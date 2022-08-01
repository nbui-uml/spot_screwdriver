import argparse
import time
import sys
from typing import Any

import cv2
import numpy as np

import bosdyn.client
import bosdyn.geometry
from bosdyn.api import image_pb2, geometry_pb2, manipulation_api_pb2
from bosdyn.client import math_helpers
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient

from screwdriver_orientation_classifier import OrientationClassifier
from spot_arm_client import ArmClient
from util import format_spotImage_to_cv2


class Screw:
    def __init__(self, rect: tuple, angle: float) -> None:
        self.x, self.y, self.w, self.h = rect
        self.angle = angle

class BoxClient:
    def __init__(self, config: argparse.Namespace, robot: bosdyn.client.Robot) -> None:
        self.config = config
        self.robot = robot


    def walk_to_box(self, pixel_x: int, pixel_y: int, image: image_pb2.ImageResponse) -> None:
        """
        Commands Spot to walk to a position 0.5m away from the box.

        Parameters
        -----
        pixel_x: int
            x position in the image.
        pixel_y: int
            y position in the image.
        image: ImageResponse
            Spot ImageResponse
        """
        robot = self.robot
        manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

        vec = geometry_pb2.Vec2(x=pixel_x, y=pixel_y)
        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole, offset_distance=0.5
        )

        walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
        cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=walk_to_request)

        while True:
            time.sleep(0.25)
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                  manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                break
        robot.logger.info('Finished.')


    def get_box_orientation(self, image: image_pb2.ImageResponse, corners: tuple) -> Any:
        """
        Get the orientation of the box from the detected corners of the AR marker.

        Parameters
        -----
        image: ImageResponse
            ImageResponse where the box was detected.
        corners: tuple(x,y)
            topLeft, topRight, bottomRight, bottomLeft
        
        Returns
        -----
        SE3Pose
            Pose of the orientation of the center of the box.
        ndarray
            XYZ unit approach vector
        """
        topLeft, topRight, bottomRight, bottomLeft = corners
        center_x = (topLeft[0] + topRight[0] + bottomRight[0] + bottomLeft[0]) // 4
        center_y = (topLeft[1] + topRight[1] + bottomRight[1] + bottomLeft[1]) // 4
        TL_x, TL_y, TL_z = bosdyn.client.image.pixel_to_camera_space(image, topLeft[0], topLeft[1])
        BL_x, BL_y, BL_z = bosdyn.client.image.pixel_to_camera_space(image, bottomLeft[0], bottomLeft[1])
        BR_x, BR_y, BR_z = bosdyn.client.image.pixel_to_camera_space(image, bottomRight[0], bottomRight[1])
        C_x, C_y, C_z = bosdyn.client.image.pixel_to_camera_space(image, center_x, center_y)

        u = np.array([TL_x - BL_x, TL_y - BL_x, TL_z - BL_z])
        v = np.array([BR_x - BL_x, BR_y - BL_x, BR_z - BL_z])
        w = np.cross(u, v)
        w = w / np.linalg.norm(w)
    
        R = 0
        P = np.arctan2(w[2], w[1])
        Y = np.arctan2(w[2], w[0])

        q = bosdyn.geometry.EulerZXY(yaw=Y, roll=R, pitch=P).to_quaternion()
        return geometry_pb2.SE3Pose(C_x, C_y, C_z, q), w


    def gaze_for_marker(self, ID: int, dict_name: str = cv2.aruco.DICT_ARUCO_ORIGINAL) -> Any:
        """
        Rotate the arm while searching for the marker and returns the corners of the marker with the image.

        Parameters
        -----
        ID: int
            ID of the marker to look for.
        dict_name: str
            Name of the aruco dictionary to use.

        Returns
        -----
        tuple
            topLeft, topRight, bottomRight, bottomLeft: tuple
                    (x,y) of corner
        ImageResponse
            ImageResponse object where marker was found.
        """
        robot = self.robot
        arm_client = ArmClient(self.config, robot)
        image_client = robot.ensure_client(ImageClient.default_service_name)
        command_client = robot.ensure_client()

        sh0, sh1, el0, el1, wr0, wr1 = arm_client.joint_states["ready"]
        arm_client.joint_move(sh0, sh1, el0, el1, wr0, wr1)

        open_command = RobotCommandBuilder.claw_gripper_open_command()
        cmd_id = command_client.robot_command(open_command)
        robot.logger.info("Opening gripper...")
        time.sleep(1)

        corners = None
        image = None
        while sh0 < 3.14 * 2:
            image_quality = 90
            image_responses = image_client.get_image([build_image_request("hand_color_image", image_quality)])

            #check if images were received
            assert len(image_responses), "Unable to get images."

            image = image_responses[0]
            img = format_spotImage_to_cv2()

            corners = self.find_marker(img, ID, dict_name)

            if corners[0] == None:
                sh0 += 3.14 / 4
                arm_client.joint_move(sh0, sh1, el0, el1, wr0, wr1)
            else:
                break
        
        close_command = RobotCommandBuilder.claw_gripper_close_command()
        cmd_id = command_client.robot_command(close_command)
        robot.logger.info("Closing gripper...")
        arm_client.stow_arm()

        if corners[0] == None:
            return corners, image
        else:
            return (None, None, None, None), None


    def find_marker(self, img: cv2.Mat, ID: int, dict_name: str = cv2.aruco.DICT_ARUCO_ORIGINAL) -> tuple:
        """
        Find the marker in the image and return the corners.
        If marker not found, returns (None, None, None, None).

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
        return (None, None, None, None)


    def get_screws(self, img: cv2.Mat) -> list:
        """
        Find and classify the orientation of screws in the image.

        Parameters
        -----
        img: ndarray
            cv2 image.
        
        Returns
        -----
        list[Screw]
            A list of detected screw objects.
        """
        angle_classifier = OrientationClassifier()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (5,5), 1)
        thresh_img = cv2.Canny(gray_img, 100,100)

        cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        screws = []
        for cnt in cnts:
            approx = cv2.contourArea(cnt)
            if approx > 300 and approx < 350:
                print(approx)
                (x,y,w,h) = cv2.boundingRect(cnt)
                
                #cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 1)
                
                clip = thresh_img[y : y + h, x : x + w]
                clip = cv2.cvtColor(clip, cv2.COLOR_GRAY2BGR)
                angle = angle_classifier.get_orientation(clip)
                screws.append(Screw((x,y,w,h), angle))
                '''
                cv2.putText(img, "%.2f" %angle,
                    (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 0, 150), 1)
                print("Angle: %.2f" %angle)
                cv2.imshow('image', img)
                cv2.imshow('Binary',thresh_img)
                cv2.waitKey()
                '''
        return screws


#-----Testing-----


def main(argv):
    from spot_docking_client import DockingClient
    from spot_arm_client import ArmClient

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    
    sdk = bosdyn.client.create_standard_sdk("BoxClient")
    robot = sdk.create_robot(options)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    docking_client = DockingClient(options, robot)
    box_client = BoxClient(options, robot)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    try:
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            docking_client.start()

            corners, image = box_client.gaze_for_marker(269)
            box_pose, approach = box_client.get_box_orientation(image, corners)

            #walk to box?
            topLeft, topRight, bottomRight, bottomLeft = corners
            center_x = (topLeft[0] + topRight[0] + bottomRight[0] + bottomLeft[0]) // 4
            center_y = (topLeft[1] + topRight[1] + bottomRight[1] + bottomLeft[1]) // 4

            box_client.walk_to_box(center_x, center_y, image)

            input("Press any key to continue...")

            docking_client.dock(520)
            return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)