import cv2
import numpy as np
from screwdriver_orientation_classifier import OrientationClassifier

def find_marker(img: cv2.Mat, ID: int, dict_name: str = cv2.aruco.DICT_ARUCO_ORIGINAL) -> tuple:
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

angle_classifier = OrientationClassifier()

img = cv2.imread("/home/csrobot/catkin_ws/src/spot_screwdriver/test_box_detection/real_box.jpg")
img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (5,5), 1)
thresh_img = cv2.Canny(gray_img, 100,100)

cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for cnt in cnts:
    approx = cv2.contourArea(cnt)
    if approx > 300 and approx < 350:
        print(approx)
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 1)

        clip = thresh_img[y : y + h, x : x + w]
        clip = cv2.cvtColor(clip, cv2.COLOR_GRAY2BGR)
        angle = angle_classifier.get_orientation(clip)
        cv2.putText(img, "%.2f" %angle,
            (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (150, 0, 150), 1)
        print("Angle: %.2f" %angle)

cv2.imshow('image', img)
cv2.imshow('Binary',thresh_img)
cv2.waitKey()