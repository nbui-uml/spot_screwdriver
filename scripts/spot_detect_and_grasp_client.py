import argparse
import sys
import os
import time

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2, gripper_camera_param_pb2
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
        Config parameters:
            bosdyn.client.util base arguments
            config.net: location of YOLOv5 ONNX file
        @param robot: type bosdyn.client.Robot, the robot instance.
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
            #format image to cv2
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
            else:
                dtype = np.uint8
            img = np.fromstring(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)
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


    def grasp_from_image(self, image: image_pb2.Image, pixel_x: int, pixel_y: int, clip: np.ndarray) -> None:
        """
        Attempts a grasp at a position defined in the image.
        @param image: type image_pb2.Image, image from Image service.
        @param pixel_x, pixel_y: type int, positions defined in image.
        """
        robot = self.robot

        gripper_camera_param_client = robot.ensure_client(GripperCameraParamClient.default_service_name)

        x, y, z = bosdyn.client.image.pixel_to_camera_space(image, pixel_x, pixel_y)
        print(f"Object at {(x, y, z)} to camera frame")
        
        #bring arm above object
        self.arm_to_above_object(image, pixel_x, pixel_y)

        #illuminate object and container
        gripper_camera_brightness = wrappers_pb2.FloatValue(0.75)
        gripper_camera_params = gripper_camera_param_pb2.GripperCameraParams(brightness=gripper_camera_brightness)
        gripper_camera_param_request = gripper_camera_param_pb2.GripperCameraGetParamRequest(params=gripper_camera_params)
        gripper_camera_param_response = gripper_camera_param_client.set_camera_params(gripper_camera_param_request)

        time.sleep(2)

        #get gripper image

            
    def arm_to_above_object(self, image: image_pb2.Image, pixel_x: int, pixel_y: int) -> None:
        """
        Move arm above a point in the image.
        @param image: type image_pb2.Image, image from Image service.
        @param pixel_x, pixel_y: type int, positions defined in image.
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
        """
        
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result


#-----Testing-----


def main(argv):
    """Command line interface."""
    from scripts.spot_docking_client import DockingClient
    from scripts.spot_arm_client import ArmClient

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
