import cv2
import numpy as np

#current marker id: 269
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


img = cv2.imread("/home/csrobot/catkin_ws/src/spot_screwdriver/test_box_detection/box2.jpg")
img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cpy = img.copy()
cpy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


aruco_params = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

if len(corners) > 0:
    ids = ids.flatten()
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4,2))
        corners = corners.astype('int')
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        cv2.line(cpy, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(cpy, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(cpy, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(cpy, bottomLeft, topLeft, (0, 255, 0), 2)

        cv2.putText(cpy, str(markerID),
            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (150, 0, 150), 2)
        print("[INFO] ArUco marker ID: {}".format(markerID))

cv2.imshow("Image", cpy)
cv2.waitKey(0)
cv2.destroyAllWindows()