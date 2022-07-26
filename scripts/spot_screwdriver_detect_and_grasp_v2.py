import argparse
import sys
import time
from typing import Any

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers, get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient

class DetectAndGraspClient:
    def __init__(self, config: argparse.Namespace):
        """
        Intializes DetectAndGraspClient instance.
        Config parameters:
            bosdyn.client.util base arguments
            config.net: location of YOLOv5 ONNX file
        """
        #robot stuff
        bosdyn.client.util.setup_logging(config.verbose)
        self.sdk = bosdyn.client.create_standard_sdk("GraspingClient")
        self.robot = self.sdk.create_robot(config.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()
        
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)

        self.image_sources = [
            "frontleft_fisheye_image",
            "frontright_fisheye_image",
            "left_fisheye_image",
            "right_fisheye_image",
            "back_fisheye_image"
        ]

        #detection model stuff
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.SCORE_THRESHOLD = 0.3
        self.NMS_THRESHOLD = 0.4
        self.CONFIDENCE_THRESHOLD = 0.5
        self.net = cv2.dnn.readNet(config.net)


    def detect_and_grasp(self) -> bool:
        """
        Iterates through the camera searching for a screwdriver. Will attempt to grasp the screwdriver if one is found.
        Returns True if successful, False otherwise.
        """
        
        robot = self.robot

        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."
        assert robot.has_arm(), "Robot requires an arm to run this client."

        #power on
        if not robot.is_powered_on():
            robot.logger.info("Powering on robot... This may take a several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")
        
        #stand
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        robot.logger.info("Getting images")
        image_responses = self.image_client.get_image_from_sources(self.image_sources)

        if len(image_responses) == 0:
            print("Unable to get images")
            assert False
        
        for image in image_responses:
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
            else:
                dtype = np.uint8
            img = np.fromstring(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)

            formatted_img = self.format_yolov5(img)
            out = self.detect(formatted_img, self.net)
            class_ids, confidences, boxes = self.wrap_detection(formatted_img, out[0])

            if len(boxes) == 1:
                box = boxes[0]
                cx = box[0] + (box[2] * 0.5)
                cy = box[1] + (box[3] * 0.5)
                self.grasp_from_image(image, cx, cy)
                return True

        return False


    def grasp_from_image(self, image: image_pb2.Image, pixel_x: int, pixel_y: int) -> None:
        """
        Attempts a grasp at a position defined in the image.
        @param image: type image_pb2.Image, image from Image service.
        @param pixel_x, pixel_y: type int, positions defined in image.
        """
        self.arm_to_above_object(image, pixel_x, pixel_y)
        self.get_object_orientation()

    
    def get_object_orientation(self):
        robot = self.robot

        assert robot.has_arm(), "Robot requires an arm to run this client."

        image_responses = self.image_client.get_image_from_sources(["hand_color_image"])

        if len(image_responses) == 1:
            image = image_responses[0]
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
            else:
                dtype = np.uint8
            img = np.fromstring(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)

            img = cv2.GaussianBlur(img, ksize=3)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_gray = np.array([0,0,0])
            upper_gray = np.array([255,50,180])
            mask = cv2.inRange(hsv, lower_gray, upper_gray)
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            res = cv2.bitwise_and(img, img, mask=mask)

            #for debugging
            cv2.imwrite("out/res.jpg", res)

            lines = cv2.HoughLinesP(res, 1, np.pi/180, 30, minLineLength=30, maxLineGap=5)

            longest = (0, None)
            if len(lines) != 0:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(res, (x1, y1), (x2, y2), (0,0,255))
                    print(f"[{(x1, y1)}, {(x2, y2)}]")
                    norm = np.linalg.norm(np.array(x1-x2, y1-y2))
                    if norm > longest[0]:
                        longest = (norm, line[0])
            else:
                assert False, "No lines detected."
            cv2.imwrite("out/lines.jpg", res)

            longest = np.array(longest[1])
            center = np.array(res.shape) * 0.5

            c1 = np.array(longest[:2]) - center
            c2 = np.array(longest[2:]) - center
            #convert to polar
            d1 = np.linalg.norm(c1)
            d2 = np.linalg.norm(c2)

            if d1 > d2:
                angle = np.arctan2(c1[1] - c2[1], c1[0] - c2[0])
            else: 
                angle = np.arctan2(c1[1] - c2[1], c1[0] - c2[0])

            print(angle)
            

    def arm_to_above_object(self, image, pixel_x, pixel_y):
        """
        Move arm above a point in the image.
        @param image: type image_pb2.Image, image from Image service.
        @param pixel_x, pixel_y: type int, positions defined in image.
        """
        robot = self.robot
        #check estop here
        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."
        assert robot.has_arm(), "Robot requires an arm to run this client."

        #power on
        if not robot.is_powered_on():
            robot.logger.info("Powering on robot... This may take a several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")
        
        #stand
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        x, y, z = bosdyn.client.image.pixel_to_camera_space(image, pixel_x, pixel_y)
        z += 0.2

        hand_ewrt_cam = geometry_pb2.Vec3(x=x, y=y, z=z)
        cam_Q_hand = geometry_pb2.Quaternion(0,0,0,1)
        cam_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_cam, rotation=cam_Q_hand)

        body_T_cam = get_a_tform_b(image.shot.transforms_snapshot,
                GRAV_ALIGNED_BODY_FRAME_NAME, image.shot.frame_name_image_sensor)
        body_T_hand = body_T_cam * math_helpers.SE3Pose.from_obj(cam_T_hand)

        arm_command = RobotCommandBuilder.arm_pose_command(
            body_T_hand.x, body_T_hand.y, body_T_hand.z,
            0.707, 0, 0, -0.707, GRAV_ALIGNED_BODY_FRAME_NAME
        )

        cmd_id = command_client.robot_command(arm_command)
        robot.logger.info("Moving arm above screwdriver.")

        block_until_arm_arrives(command_client, cmd_id, 10)
        robot.logger.info("Finish moving above screwdriver.")
    

    def detect(self, image: np.ndarray, net: cv2.dnn.Net) -> np.ndarray:
        """
        Puts the image through the DNN
        """
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), crop=False)
        net.setInput(blob)
        preds = net.forward()
        return preds


    def wrap_detection(self, input_image: np.ndarray, output_data: np.ndarray) -> tuple:
        """
        Parses output data from DNN and returns result_class_ids, result_confidences, result_boxes.
        """
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= self.CONFIDENCE_THRESHOLD:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes


    def format_yolov5(self, frame: np.ndarray) -> np.ndarray:
        """
        Pads image for YOLOv5.
        """
        row, col = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result