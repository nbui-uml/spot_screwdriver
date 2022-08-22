import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
from bosdyn.api import image_pb2
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.image import ImageClient, build_image_request


def capture_gripper_image(robot:bosdyn.client.Robot, fn: str) -> None:
    """
    Captures and saves a gripper image.

    Parameters
    -----
    robot: Robot
        Robot instance.
    fn: str
        Name of file to save to.
    """
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    open_command = RobotCommandBuilder.claw_gripper_open_command()
    cmd_id = command_client.robot_command(open_command)

    image_responses = image_client.get_image([build_image_request("hand_color_image")])

    #check if images were received
    assert len(image_responses) == 1, "Unable to get valid images."

    close_command = RobotCommandBuilder.claw_gripper_close_command()
    cmd_id = command_client.robot_command(close_command)

    image = image_responses[0]
    img = format_spotImage_to_cv2(image)
    cv2.imwrite(fn, img)
    print("Wrote image to %s" %fn)


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


def get_median_depth(depth_img: np.ndarray, pixel_x: int, pixel_y: int, range: int=6) -> int:
    """
    Returns the raw median depth value in a range around the point in the image.

    Parameters
    -----
    depth_img: ndarray
        Depth image as numpy array.
    pixel_x: int
        x position in the image.
    pixel_y: int
        y position in the image.
    range: int
        The size of the square to search in.

    Returns
    -----
    int
        The raw median depth value (mm).
    """
    depth_array = depth_img[pixel_y-range//2:pixel_y+range//2, pixel_x-range//2:pixel_x+range//2]
    depth_array = depth_array.flatten()
    nonzero_array = []
    for i in depth_array:
        if i != 0.0:
            nonzero_array.append(i)
    nonzero_array.sort()
    if len(nonzero_array) == 0:
        return 0.0
    else:
        return nonzero_array[len(nonzero_array)//2]


def pixel_to_camera_frame(visual_image: image_pb2.ImageResponse, depth_image: image_pb2.ImageResponse, pixel_x: int, pixel_y: int) -> tuple:
    """
    Calculates the world position of the pixel in reference to camera frame.

    Parameters
    -----
    visual_image: ImageResponse
        ImageResponse for the visual image.
    depth_image: ImageResponse
        ImageResponse for the depth image in the visual frame (_depth_in_visual_frame).
    pixel_x: int
        x position in the image.
    pixel_y: int
        y position in the image.
    
    Returns
    -----
    tuple
        x, y, z (in meters) in reference to camera frame.
    """
    depth_img = np.frombuffer(depth_image.shot.image.data, dtype=np.uint16)
    depth_img = depth_img.reshape(depth_image.shot.image.rows, depth_image.shot.image.cols)
    z = get_median_depth(depth_img, pixel_x, pixel_y) / depth_image.source.depth_scale
    x, y, z = bosdyn.client.image.pixel_to_camera_space(visual_image, pixel_x, pixel_y, z)
    return x, y, z


def reverse_rotation_map(org_img: cv2.Mat, rot_img: cv2.Mat, angle: float) -> cv2.Mat:
    """
    Reverses a rotation using the original and rotated image.

    Parameters
    -----
    org_img: ndarray
        cv2 image object of original image.
    rot_img: ndarray
        cv2 image object of rotated image.
    angle: float
        Rotation angle.
    
    Returns
    -----
    ndarray
        Image at the original orientation.
    """
    mapped = ndimage.rotate(rot_img, -angle)
    rot_h, rot_w, _ = mapped.shape
    org_h, org_w, _ = org_img.shape
    return mapped[(rot_h - org_h)//2:(rot_h + org_h)//2, (rot_w - org_w)//2:(rot_w - org_w)//2]