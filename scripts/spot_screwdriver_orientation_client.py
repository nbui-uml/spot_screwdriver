import argparse

import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import math_helpers
from bosdyn.api import image_pb2, geometry_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import get_a_tform_b, HAND_FRAME_NAME


from util import format_spotImage_to_cv2
from screwdriver_orientation_classifier import OrientationClassifier

SCREWDRIVER_FRAME_NAME = "screwdriver"

class ScrewdriverOrientationClient:
    def __init__(self, config: argparse.Namespace, robot: bosdyn.client.Robot, cascade: str = None) -> None:
        """
        Initializes ScrewdriverOrientationClient instance.
        """
        self.robot = robot
        self.config = config

        if cascade is None:
            cascade = "/home/csrobot/catkin_ws/src/spot_screwdriver/models/screwdriver_front_cascade.xml"
        self.screwdriver_locator = cv2.CascadeClassifier(cascade)

        self.ROTATION_ANGLE = {
            'frontleft_fisheye_image': -78,
            'frontright_fisheye_image': -102,
        }

        self.hand_T_screwdriver = None

    
    def get_orientation_from_camera(self, camera_name: str = "frontleft_fisheye_image") -> float:
        """
        Get the orientation of the screwdriver from a camera.

        Parameters
        -----
        camera_name: str
            Name of the camera to use. Defaults to 'frontleft_fisheye_image'.

        Returns
        -----
        float
            Angle of the screwdriver. Returns 4.0 if unable to determine angle.
        """
        robot = self.robot

        image_client = robot.ensure_client(ImageClient.default_service_name)

        image_request = [build_image_request(camera_name)]
        image_responses = image_client.get_image(image_request)

        if len(image_responses) == 1:
            image = image_responses[0]
            img = format_spotImage_to_cv2(image)
            img = ndimage.rotate(img, self.ROTATION_ANGLE[camera_name])
            cv2.imwrite("out/img.jpg", img)
            angle = self.angle_from_raw_image(img)
            if angle == 4.0:
                print("Unable to determine angle.")
            return angle
        else:
            print("Unable to get image.")
            return 4.0
    

    def angle_from_raw_image(self, img: cv2.Mat) -> float:
        """
        Calculates the angle from the camera data.

        Parameters
        -----
        img: ndarray 
            Image data from camera

        Returns
        -----
        float
            Calculated angle of screwdriver. Returns 4.0 if unable to calculate angle.
        """
        angle_classifer = OrientationClassifier()

        #detect screwdriver
        height, width = img.shape
        scale_factor = 800 / height
        img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
        img = img[height//3:2*(height//3), width//3:2*(width//3)]

        found = self.screwdriver_locator.detectMultiScale(img, minNeighbors=5, minSize=(50,50), maxSize=(250,250))
        print(f"Cascade applied. Found {len(found)}")

        #if valid, focus in on screwdriver
        if len(found) == 1:
            #resize to square for orientation classifier
            x, y, w, h = found[0]

            l = (w + h) // 2
            print(f"l: {l}, w: {w}, h: {h}")
            c = (x + (w // 2), y + (h // 2))
            x, y = (c[0] - (l // 2), c[1] - (l // 2))

            #debugging
            cpy = img.copy()
            cv2.rectangle(cpy, (x,y), (x+w, y+h), (0,0,5))
            cv2.imwrite("out/headon.jpg", cpy)

            img_cropped = img[y:y+l, x:x+l]
            img_cropped = cv2.flip(img_cropped, 1) #mirror
            #pass through classifier and return value
            return angle_classifer.get_orientation(img_cropped)
        #else return dummy value
        else:
            return 4.0

    
    def save_hand_T_screwdriver(self, angle: float) -> None:
        """
        Saves the transform from the hand to the screwdriver.

        Parameters
        -----
        angle: float
            Angle of the screwdriver in the hand in radians.
        """
        robot = self.robot

        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        screwdriver_position = geometry_pb2.Vec3(0.08,0,0)
        q = math_helpers.Quat.from_roll(angle)
        self.hand_T_screwdriver = geometry_pb2.SE3Pose(position=screwdriver_position, rotation=q)


    def get_frame_T_screwdriver(self, parent_frame: str, tf: geometry_pb2.FrameTreeSnapshot) -> math_helpers.SE3Pose:
        """
        Gets the transform from the given frame to the screwdriver.

        Parameters
        -----
        parent_frame: str
            Name of the parent frame.
        tf: FrameTreeSnapshot
            FrameTreeSnapshot that contains the parent frame.
        
        Returns
        -----
        SE3Pose
            Transform from parent_frame to screwdriver.
        """
        if self.hand_T_screwdriver == None:
            print("No screwdriver transform recorded.")
        frame_T_hand = get_a_tform_b(tf, parent_frame, HAND_FRAME_NAME)
        return frame_T_hand * math_helpers.SE3Pose.from_obj(self.hand_T_screwdriver)
