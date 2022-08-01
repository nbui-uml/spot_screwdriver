import cv2
import numpy as np

import bosdyn.client
from bosdyn.api import image_pb2



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