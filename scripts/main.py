import argparse
import sys
import math
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient

import spot_docking

from util import *

def main(argv):
    #initialize Boston Dynamics client
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--net", dest="net", default="/home/csrobot/catkin_ws/src/spot_screwdriver/models/screwdriver_yolo5.onnx")
    options = parser.parse_args(argv)
    bosdyn_logger = bosdyn.client.util.get_logger()

    #Create and authenticate robot
    sdk = bosdyn.client.create_standard_sdk("SpotScrewdriverClinet")
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    try:
        lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    except Exception as e:
        bosdyn_logger.exception("Could not initialize client.")
        return False

    try:
        #walk to box (user control)
        print("Use the tablet control to navigate towards the box. Press any key when finished...")
        print("Taking control in 3 seconds...")
        time.sleep(3)

        #analyze box (program control = take lease)
        lease_client.take()
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            pass

        #walk to screwdriver (user control)
        print("Use the tablet control to navigate to the screwdriver. Press any key when finished...")
        print("Taking control in 3 seconds...")
        time.sleep(3)

        #pick up screwdriver and operate on box (program control = take lease)
        lease_client.take()
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            pass
    except Exception as e:
        bosdyn_logger.exception("Exception thrown.")
        print(e)
        return False


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)