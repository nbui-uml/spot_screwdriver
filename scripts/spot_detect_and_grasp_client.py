import argparse
import sys
import os
import time
from typing import Any

import cv2
from matplotlib.pyplot import close
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2, gripper_camera_param_pb2, gripper_command_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers, get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from google.protobuf import wrappers_pb2

from spot_arm_client import ArmClient


class DetectAndGraspClient:
    def __init__(self, config: argparse.Namespace, robot: bosdyn.client.Robot) -> None:
        """
        Intializes DetectAndGraspClient instance.

        Parameters
        -----
        config: Namespace
            bosdyn.client.util base arguments
            config.net: str
                Location of YOLOv5 ONNX file.
        robot: Robot
            The robot instance.
        """
        #robot stuff
        self.config = config
        self.sdk = bosdyn.client.create_standard_sdk("GraspingClient")
        self.robot = robot

        self.image_sources = [
            "frontleft_fisheye_image",
            "frontright_fisheye_image"
        ]

        #detection model stuff
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.SCORE_THRESHOLD = 0.25
        self.NMS_THRESHOLD = 0.45
        self.CONFIDENCE_THRESHOLD = 0.5
        self.net = cv2.dnn.readNet(config.net)


    def detect_and_grasp(self) -> bool:
        """
        Iterates through the camera searching for a screwdriver. Will attempt to grasp the screwdriver if one is found.
        Returns True if successful, False otherwise.

        Returns
        -----
        bool
        """
        
        robot = self.robot

        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."
        assert robot.has_arm(), "Robot requires an arm to run this client."

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        image_client = robot.ensure_client(ImageClient.default_service_name)

        #power on
        if not robot.is_powered_on():
            robot.logger.info("Powering on robot... This may take a several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")
        
        #stand
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        robot.logger.info("Getting images")
        image_quality = 75
        image_responses = image_client.get_image([build_image_request(source, image_quality) for source in self.image_sources])

        #check if images were received
        assert len(image_responses), "Unable to get images."
        
        for image in image_responses:
            img = self.format_spotImage_to_cv2(image)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            #send image through detection model
            formatted_img = self.format_yolov5(img)
            out = self.detect(formatted_img, self.net)
            class_ids, confidences, boxes = self.wrap_detection(formatted_img, out[0])

            #check for detections
            if len(boxes) == 1:
                box = boxes[0]
                if box[2] > 300 or box[3] > 300:
                    print(f"No objects found in {image.shot.frame_name_image_sensor}")
                    continue

                cx = int(box[0] + (box[2] * 0.5))
                cy = int(box[1] + (box[3] * 0.5))
                print(f"Object found at {(cx, cy)}, {box} in {image.shot.frame_name_image_sensor}")

                if box[0] < 0:
                    box[0] = 0
                if box[1] < 0:
                    box[1] = 0
                if box[0] + box[2] > img.shape[1]:
                    box[2] = img.shape[1] - box[0]
                if box[1] + box[3] > img.shape[0]:
                    box[3] = img.shape[0] - box[1]
                
                #debugging
                if os.path.exists("out/detect.jpg"):
                    os.remove("out/detect.jpg")
                boxed_img = img.copy()
                cv2.rectangle(boxed_img, box, (0,0,255), 2)
                cv2.imwrite("out/detect.jpg", boxed_img)
                
                clip = img[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]

                self.grasp_from_image(image, cx, cy, clip)
                return True
            else:
                print(f"No objects found in {image.shot.frame_name_image_sensor}")

        return False


    def grasp_from_image(self, image: image_pb2.ImageResponse, pixel_x: int, pixel_y: int) -> None:
        """
        Attempts a grasp at a position defined in the image.

        Parameters
        -----
            image: image_pb2.ImageResponse
                ImageResponse from Image service.
            pixel_x: int
                x position in image.
            pixel_y: int
                y position in image.

        Returns
        -----
        None
        """
        robot = self.robot
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        x, y, z = bosdyn.client.image.pixel_to_camera_space(image, pixel_x, pixel_y)
        print(f"Object at {(x, y, z)} to camera frame")
        
        #bring arm above object
        self.arm_to_above_object(image, pixel_x, pixel_y)

        x, y, z, image = self.position_on_gripper_camera()
        
        arm_client = ArmClient(self.config, robot)
        arm_client.arm_move(x, y, z - 0.2, 1, 0, 0, 0, image.shot.frame_name_image_sensor, image.shot.transforms_snapeshot)

        open_command = RobotCommandBuilder.claw_gripper_open_command()
        cmd_id = command_client.robot_command(open_command)
        robot.logger.info("Opening claw...")
        time.sleep(1)

        arm_client.arm_move(x, y, z, 1, 0, 0, 0, image.shot.frame_name_image_sensor, image.shot.transforms_snapeshot)

        close_command = RobotCommandBuilder.claw_gripper_close_command()
        cmd_id = command_client.robot_command(close_command)
        robot.logger.info("Grasping.")

        arm_client.arm_move(x, y, z - 0.4, 1, 0, 0, 0, image.shot.frame_name_image_sensor, image.shot.transforms_snapeshot)
        

    def position_on_gripper_camera(self) -> Any:
        """
        Calculates the position of the screwdriver in relation to the gripper camera.
        
        Returns
        ------
        tuple
            x: float
            y: float
            z: float
            image: image_pb2.ImageResponse
        """
        robot = self.robot

        gripper_camera_param_client = robot.ensure_client(GripperCameraParamClient.default_service_name)
        image_client = robot.ensure_client(ImageClient.default_service_name)
        
        #illuminate object and container
        gripper_camera_brightness = wrappers_pb2.FloatValue(0.75)
        gripper_camera_params = gripper_camera_param_pb2.GripperCameraParams(brightness=gripper_camera_brightness)
        gripper_camera_param_request = gripper_camera_param_pb2.GripperCameraGetParamRequest(params=gripper_camera_params)
        gripper_camera_param_response = gripper_camera_param_client.set_camera_params(gripper_camera_param_request)

        time.sleep(2)

        #get gripper image
        robot.logger.info("Getting images")
        image_quality = 75
        image_responses = image_client.get_image([build_image_request("hand_color_image", image_quality)])

        #check if images were received
        assert len(image_responses) == 1, "Unable to get valid images."

        image = image_responses[0]
        img = self.format_spotImage_to_cv2(image)

        #analysis algorithm
        img = cv2.GaussianBlur(img, (5,5), 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([0,0,100])
        upper_range = np.array([255,50,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.bitwise_not(mask)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=100, maxRadius=500)

        clip = None
        clipy = None
        clipx = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                #cv2.circle(img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(img, center, radius, (255, 0, 255), 3)
                radius = radius // 2
                clip = mask[center[1] - radius : center[1] + radius, center[0] - radius : center[0] + radius]
                clipy = center[1] - radius
                clipx = center[0] - radius

        averages = np.zeros((4,4))
        hstep = clip.shape[0] // 4
        wstep = clip.shape[1] // 4
        for y in range(4):
            for x in range(4):
                section = clip[y * hstep : (y+1) * hstep, x * wstep : (x+1) * wstep]
                averages[y][x] = np.average(section)

        zones = np.zeros((3,3))
        for y in range(3):
            for x in range(3):
                zones[y][x] = np.sum(averages[y : y + 1, x : x + 1])

        i,j = np.unravel_index(zones.argmax(), zones.shape)
        pixel_y = clipy + int((i + 0.5) * (clip.shape[0] // 3))
        pixel_x = clipx + int((j + 0.5) * (clip.shape[1] // 3))

        x, y, z = bosdyn.client.image.pixel_to_camera_space(image, pixel_x, pixel_y)

        return x, y, z, image
            

    def arm_to_above_object(self, image: image_pb2.ImageResponse, pixel_x: int, pixel_y: int) -> None:
        """
        Move arm above a point in the image.

        Parameters
        -----
        image: ImageResponse
            Image from Image service.
        pixel_x: int
            X position in image.
        pixel_y: int
            Y position in image.
        """
        robot = self.robot
        arm_client = ArmClient(self.config, robot)

        x, y, z = bosdyn.client.image.pixel_to_camera_space(image, pixel_x, pixel_y)
        y += 0.2

        cam_T_body = get_a_tform_b(image.shot.transforms_snapshot,
                image.shot.frame_name_image_sensor, GRAV_ALIGNED_BODY_FRAME_NAME)
        hand_T_body = cam_T_body * math_helpers.SE3Pose(0,0,0, geometry_pb2.Quaternion(0.707, 0, 0, -0.707))

        arm_client.arm_move(
            x, y, z, 
            hand_T_body.rot.x, hand_T_body.rot.y, hand_T_body.rot.z, hand_T_body.rot.w,
            image.shot.frame_name_image_sensor, image.shot.transforms_snapshot
        )


    def detect(self, image: np.ndarray, net: cv2.dnn.Net) -> np.ndarray:
        """
        Puts the image through the DNN.

        Parameters
        -----
        image: ndarry
            Formatted image for DNN.
        net: dnn.Net
            DNN net object.
        
        Returns
        -----
        ndarry
            Result of detection model.
        """
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), crop=False)
        net.setInput(blob)
        preds = net.forward()
        return preds


    def wrap_detection(self, input_image: np.ndarray, output_data: np.ndarray) -> tuple:
        """
        Parses output data from DNN.
        
        Parameters
        -----
        input_image: ndarray
            Original cv2 image.
        output_data: ndarray
            Output from DNN.

        Returns
        -----
        tuple
            result_class_ids: list 
            result_confidences: list
            result_boxes: list
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
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.SCORE_THRESHOLD, self.NMS_THRESHOLD) 

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

        Parameters
        -----
        frame: ndarray
            Raw cv2 image.
        
        Returns
        -----
        ndarray
            Formatted cv2 image for YOLOv5 model.
        """ 
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def format_spotImage_to_cv2(image: image_pb2.ImageResponse) -> cv2.Mat:
        """
        Format Spot Image to cv2.

        Parameters
        -----
        image: ImageResponse
            Image from Spot
        
        Returns
        -----
        ndarray
            cv2 formatted image.
        """
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)
        return img

#-----Testing-----


def main(argv):
    """Command line interface."""
    from spot_docking_client import DockingClient
    from spot_arm_client import ArmClient

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--net", help="Path to yolov5 ONNX file",  
        default="/home/csrobot/catkin_ws/src/spot_screwdriver/models/screwdriver_yolo5.onnx")
    options = parser.parse_args(argv)
    #try: #get lease here and do stuff
        #docking_client = DockingClient(options)
    grasping_client = DetectAndGraspClient(options)
        #arm_client = ArmClient(options)
    lease_client = grasping_client.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        #with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            #undock
            #docking_client.undock()

    input("Press any key to continue")

    while True:
        success = grasping_client.detect_and_grasp()
        if success == True:
            print("Screwdriver found")
        else:
            print("Unable to find screwdriver")
        if input("Try again? (y/n)").lower() == "n":
            break

            #arm_client.stow_arm()

            #dock
            #docking_client.dock(520)
        #return True
        """
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False
        """

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
