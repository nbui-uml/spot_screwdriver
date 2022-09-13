import argparse
import sys
import time

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient

import spot_docking
import util

def find_marker(img: cv2.Mat, ID: int, dict_name: str = cv2.aruco.DICT_ARUCO_ORIGINAL) -> tuple:
    """
    Find the marker in the image and return the corners.
    If marker not found, returns None.

    Parameters
    -----
    img: ndarray
        cv2 image.
    ID: int
        ID of the marker to look for
    dict_name: str
        Name of the aruco dictionary to use.

    Returns
    -----
    tuple
        topLeft, topRight, bottomRight, bottomLeft: tuple
            (x,y) of corner
    """
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            if markerID == ID:
                corners = markerCorner.reshape((4,2))
                corners = corners.astype('int')
                return corners
    return None


def get_alignment_vector_to_vision_frame(robot, id):
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_cleint(RobotCommandClient.default_service_name)

    open_command = RobotCommandBuilder.claw_gripper_open_command()
    cmd_id = command_client.robot_command(open_command)
    robot.logger.info("Opening gripper...")
    time.sleep(1)

    sources = ["hand_depth_in_hand_color_frame", "hand_color_image"]
    image_responses = image_client.get_image_from_sources(sources)

    #check if images were received
    assert len(image_responses) == 2, "Unable to get images."

    close_command = RobotCommandBuilder.claw_gripper_close_command()
    cmd_id = command_client.robot_command(close_command)
    robot.logger.info("Closing gripper...")

    depth = image_responses[0]
    image = image_responses[1]
    img = util.format_spotImage_to_cv2(image)
    TL, TR, BR, BL = find_marker(img, id)

    BL_x, BL_y, BL_z = util.pixel_to_camera_frame(image, depth, BL[0], BL[1])
    BR_x, BR_y, BR_z = util.pixel_to_camera_frame(image, depth, BR[0], BR[1])

def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    options = parser.parse_args(argv)

    bosdyn.client.util.setup_logging(options.verbose)
    sdk = bosdyn.client.create_standard_sdk("BoxManipulationClient")
    robot = sdk. create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot needs an arm lmao"
    assert not robot.is_estopped, "Robot is estopped."

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        spot_docking.start()
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        #put arm in ready position (unstow)
        unstow = RobotCommandBuilder.arm_ready_command()
        cmd_id = command_client.robot_command(unstow)
        robot.logger.info("Unstowing arm...")
        block_until_arm_arrives(command_client, cmd_id, 3)
        #get alignment vector from AR marker
        #create grasp constraint
        #create grasp command
        #grasp the door

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)