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
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
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


    def grasp_from_image(self, image: image_pb2.Image, x: int, y: int) -> None:
        """
        Attempts a grasp at a position defined in the image.
        @param image: type image_pb2.Image, image from Image service.
        @param x, y: type int, positions defined in image.
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

        #construct commmand
        robot.logger.info(f"Picking object at image location ({x},{y})")
        pick_vec = geometry_pb2.Vec2(x=x, y=y)
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor, camera_model=image.source.pinhole
        )

        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        #send request
        cmd_response = self.manipulation_api_client.manipulation_api_command(manipulation_api_request=grasp_request)

        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = self.manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                  manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            time.sleep(0.25)

        robot.logger.info('Finished grasp.')


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
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result


#-----Testing-----

def main(argv):
    """Command line interface."""
    from spot_dock_undock import DockingClient
    from spot_screwdriver_orientation import ArmClient

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--net", help="Path to yolov5 ONNX file",  
        default="/home/csrobot/catkin_ws/src/spot_screwdriver/models/screwdriver_yolo5.onnx")
    options = parser.parse_args(argv)
    try: #get lease here and do stuff
        docking_client = DockingClient(options)
        grasping_client = DetectAndGraspClient(options)
        arm_client = ArmClient(options)
        lease_client = grasping_client.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            #undock
            docking_client.undock()

            success = grasping_client.detect_and_grasp()
            if success == True:
                print("Grasp successful")
            else:
                print("Unable to find and grasp screwdriver")

            arm_client.stow_arm()

            #dock
            docking_client.dock(520)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)