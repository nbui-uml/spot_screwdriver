import math
import argparse
import sys
from urllib.parse import non_hierarchical
import cv2
import numpy as np
import os
from scipy import ndimage

import bosdyn.client
from bosdyn.api import geometry_pb2
from bosdyn.client.image import ImageClient, build_image_request
from spot_detect_and_grasp_client import DetectAndGraspClient
from bosdyn.client.frame_helpers import math_helpers, get_a_tform_b, BODY_FRAME_NAME, ODOM_FRAME_NAME, HAND_FRAME_NAME
from util import format_spotImage_to_cv2

def reverse_rotation_map(org_center, rot_center, rot_img, angle):
    #debugging
    out_path = f"out/{image.shot.frame_name_image_sensor}_rotated.jpg"
    if os.path.exists(out_path):
        os.remove(out_path)
    cv2.imwrite(out_path, rot_img)
    cv2.imshow("Rotated", rot_img)
    mapped = ndimage.rotate(rot_img, -angle)
    h, w, _ = mapped.shape
    mapped = mapped[h//2 - org_center[0]:h//2 + org_center[0], w//2 - org_center[1]:w//2 + org_center[1]]
    cv2.imshow("Mapped", mapped)
    cv2.waitKey(0)
    hsv_mapped = cv2.cvtColor(mapped, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 150, 150])
    higher = np.array([255, 255, 255])
    mask = cv2.inRange(hsv_mapped, lower, higher)
    return np.argwhere(mask)

def get_median_depth(depth_img, pixel_x, pixel_y, range=6):
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

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
bosdyn.client.util.add_base_arguments(parser)
parser.add_argument("--net", help="Path to yolov5 ONNX file",  
    default="/home/csrobot/catkin_ws/src/spot_screwdriver/models/screwdriver_yolo5.onnx")
options = parser.parse_args(argv)

sdk = bosdyn.client.create_standard_sdk("GraspingClient")
robot = sdk.create_robot(options.hostname)
bosdyn.client.util.authenticate(robot)
robot.time_sync.wait_for_sync()

image_client = robot.ensure_client(ImageClient.default_service_name)
detect_and_grasp_client = DetectAndGraspClient(options, robot, options.net)

cameras = ["frontleft", "frontright"]

for camera in cameras:
    sources = [camera + '_depth_in_visual_frame', camera + '_fisheye_image']
    image_responses = image_client.get_image_from_sources(sources)
    if len(image_responses) < 2:
        print(f"Failed to get images from {camera}.")
        continue
    depth_image = image_responses[0]
    '''
    depth = np.frombuffer(depth_image.shot.image.data, dtype=np.uint16)
    depth = depth.reshape(depth_image.shot.image.rows, depth_image.shot.image.cols)
    '''
    depth = format_spotImage_to_cv2(depth_image)
    image = image_responses[1]
    img = format_spotImage_to_cv2(image)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    '''
    img_cy = img.shape[0] // 2
    img_cx = img.shape[1] // 2
    img = img[img_cy - depth.shape[0] // 2:img_cy + depth.shape[0] // 2, img_cx - depth.shape[1] // 2:img_cx + depth.shape[1] // 2]
    '''
    
    min_val = np.min(depth)
    max_val = np.max(depth)
    depth_range = max_val - min_val
    depth8 = (255.0 / depth_range * (depth - min_val)).astype('uint8')
    depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
    depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
    # Add the two images together.
    out = cv2.addWeighted(img, 0.5, depth_color, 0.5, 0)
    out = ndimage.rotate(out, detect_and_grasp_client.ROTATION_ANGLE[image.shot.frame_name_image_sensor])
    cv2.imshow("Visual depth", out)
   

    cols, rows, _ = img.shape
    rotated_img = ndimage.rotate(img, detect_and_grasp_client.ROTATION_ANGLE[image.shot.frame_name_image_sensor])

    #debugging
    out_path = f"out/{image.shot.frame_name_image_sensor}.jpg"
    if os.path.exists(out_path):
        os.remove(out_path)
    cv2.imwrite(out_path, rotated_img)

    #send image through detection model
    formatted_img = detect_and_grasp_client.format_yolov5(rotated_img)
    out = detect_and_grasp_client.detect(formatted_img, detect_and_grasp_client.net)
    class_ids, confidences, boxes = detect_and_grasp_client.wrap_detection(formatted_img, out[0])

    out_path = f"out/{image.shot.frame_name_image_sensor}_detect.jpg"
    if os.path.exists(out_path):
        os.remove(out_path)
    boxed_img = rotated_img.copy()

    #check for detections
    points = []
    for box in boxes:
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
        cv2.rectangle(boxed_img, box, (0,0,255), 2)
        #cv2.circle(rotated_img, (cx, cy), 1, (0,0,255))
        rotated_img[cy][cx] = [0,0,255]
        
        points.append([cx, cy])
    cv2.imwrite(out_path, boxed_img)

    if len(points) == 0:
        valid_points = []
    else:
        rot_center = np.array([rotated_img.shape[0] // 2, rotated_img.shape[1]//2])
        transformed_points = reverse_rotation_map(np.array([cols // 2, rows //2]), rot_center, rotated_img, detect_and_grasp_client.ROTATION_ANGLE[image.shot.frame_name_image_sensor])
        print(transformed_points)

        valid_points = []
        body_T_cam = get_a_tform_b(image.shot.transforms_snapshot, BODY_FRAME_NAME, image.shot.frame_name_image_sensor)
        for point in transformed_points: #actually, how do i find the distance?????
            cv2.circle(img, (int(point[1]), int(point[0])), 3, (0,255,0), 3)
            z = get_median_depth(depth, int(point[1]), int(point[0])) / depth_image.source.depth_scale
            x, y, z = bosdyn.client.image.pixel_to_camera_space(image, point[1], point[0], z)
            if z == 0.0:
                continue
            print(x,y,z)
            cam_T_point_position = geometry_pb2.Vec3(x=x, y=y, z=z)
            cam_T_point = geometry_pb2.SE3Pose(position=cam_T_point_position, rotation=geometry_pb2.Quaternion(w=0,x=0,y=0,z=0))
            body_T_point = body_T_cam * math_helpers.SE3Pose.from_obj(cam_T_point)
            print(f"{body_T_point}")
            if body_T_point.x < 2.0:
                if math.fabs(body_T_point.y) < 1.0:
                    if math.fabs(body_T_point.z) < 1.0:
                        valid_points.append(point)

    if len(valid_points) == 0:
        print(f"No valid objects found in {image.shot.frame_name_image_sensor}")
    else:
        print(f"{len(valid_points)} valid objects found in {image.shot.frame_name_image_sensor}.")
        print(valid_points)
        for point in valid_points:
            cx = int(point[1])
            cy = int(point[0])
            cv2.circle(img, (cx, cy), 3, (0,0,255), 1)

    out_path = f"out/{image.shot.frame_name_image_sensor}_detectoriginal.jpg"
    if os.path.exists(out_path):
        os.remove(out_path)
    cv2.imwrite(out_path, img)