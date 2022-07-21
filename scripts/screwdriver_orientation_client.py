#bring screwdriver so Spot can view it head-on
#While angle not in threshold:
    #rotate screwdriver
#initialize frame for screwdriver in relation to hand

import argparse
from sre_constants import FAILURE, SUCCESS
import sys
import math

import bosdyn.client
import bosdyn.client.util

import rospy

from spot_arm_client import ArmClient
from spot_screwdriver.srv import ScrewdriverOrientationSrv, ScrewdriverOrientationSrvRequest, ScrewdriverOrientationSrvResponse

get_screwdriver_orientation = None
L_cam_topic = rospy.get_param("L_front_cam_topic", "camera/frontleft/camera/image")
R_cam_topic = rospy.get_param("R_front_came_topic", "camera/frontright/camera/image")


def main(argv) -> int:
    #initialize Boston Dynamics client
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    bosdyn_logger = bosdyn.client.util.get_logger()
    try:
        arm_client = ArmClient(options)
    except Exception as e:
        bosdyn_logger.exception("Could not initialize ArmClient")
        return FAILURE

    #intialize ROS
    rospy.init_node("screwdriver_orientation_client")
    rospy.wait_for_service("screwdriver_orientation")
    global get_screwdriver_orientation
    get_screwdriver_orientation = rospy.ServiceProxy("screwdriver_orientation", ScrewdriverOrientationSrv)

    try:
        sh0,sh1,el0,el1,wr0,wr1 = arm_client.joint_states["front"]
        arm_client.arm_to_pose(sh0,sh1,el0,el1,wr0,wr1)
    except Exception as e:
        bosdyn_logger.exception("Exception moving arm to front")
        return FAILURE

    success = False
    RIGHT_CAM = 1 #1 for right cam, -1 for left cam
    while success == False:
        #Face one of the cameras
        print(f"Moving to {'right' if RIGHT_CAM == 1 else 'left'} camera position")
        try:
            sh0,sh1,el0,el1,wr0,wr1 = arm_client.joint_states["front"]
            #something here
            arm_client.arm_to_pose(sh0,sh1,el0,el1,wr0,wr1)
            print(f"Moved to {'right' if RIGHT_CAM == 1 else 'left'} camera position")
        except Exception as e:  # pylint: disable=broad-except
            bosdyn_logger.exception("Threw an exception")
            return FAILURE
        
        #call the /screwdriver_orientation service
        print(f"Getting screwdriver orientation from {'right' if RIGHT_CAM == 1 else 'left'} camera")
        cam_topic = R_cam_topic if RIGHT_CAM == 1 else L_cam_topic
        response = get_screwdriver_orientation(topic=cam_topic)
        success = response.success
        if success == False:
            print(f"Unable to get screwdriver orientation from {'right' if RIGHT_CAM == 1 else 'left'} camera")
        else:
            print(f"Screwdriver angle: {response.angle}")
        RIGHT_CAM = RIGHT_CAM * -1
        
    try:
        arm_client.power_off_safe()
        return SUCCESS
    except Exception as e:
        bosdyn_logger.exception("Exception powering off")
        return FAILURE


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
