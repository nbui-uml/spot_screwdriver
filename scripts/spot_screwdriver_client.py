import argparse
import sys
import math
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient

from spot_arm_client import ArmClient
from spot_docking_client import DockingClient
from spot_detect_and_grasp_client import DetectAndGraspClient
from spot_screwdriver_orientation_client import ScrewdriverOrientationClient

def main(argv):
    #initialize Boston Dynamics client
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--net", dest="net", default="/home/csrobot/catkin_ws/src/spot_screwdriver/models/screwdriver_yolo5.onnx")
    options = parser.parse_args(argv)
    bosdyn_logger = bosdyn.client.util.get_logger()

    sdk = bosdyn.client.create_standard_sdk("GraspingClient")
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    try:
        arm_client = ArmClient(options, robot)
        grasping_client = DetectAndGraspClient(options, robot, options.net)
        docking_client = DockingClient(options, robot)
        lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    except Exception as e:
        bosdyn_logger.exception("Could not initialize client.")
        return False

    try:
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            docking_client.undock()
            if grasping_client.detect_and_grasp() == True:
                robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
                robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
                screwdriver_orientation_client = ScrewdriverOrientationClient()

                #move backwards
                robot_state = robot_state_client.get_robot_state()
                backwards_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(-0.3, 0, 0, robot_state.kinematic_state.transforms_snapshot)
                robot_command_client.robot_command(backwards_cmd)
                time.sleep(2)

                #look at and analyze screwdriver
                #to front of camera
                robot_state = robot_state_client.get_robot_state()
                arm_client.arm_move(0,0,0.3,0,1,0,0,"frontright_fisheye", robot_state.kinematic_state.transforms_snapshot)
                angle = screwdriver_orientation_client.get_orientation_from_camera("frontright_fisheye")
                print("Angle %.2f" %angle)

                ready = arm_client.joint_states["ready"]
                arm_client.joint_move(ready[0], ready[1], ready[2], ready[3], ready[4], ready[5], ready[6])

                time.sleep(1)

                #return object to container
                grasping_client.return_object_to_container()
            else:
                print("Unable to grasp screwdriver.")

            arm_client.stow_arm()
            docking_client.dock(520)
            return True
    except Exception as e:
        bosdyn_logger.exception("Exception thrown.")
        print(e)
        return False

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)