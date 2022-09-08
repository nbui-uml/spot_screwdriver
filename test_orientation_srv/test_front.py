import cv2
import numpy as np
from screwdriver_orientation_classifier import OrientationClassifier

screwdriver_locator = cv2.CascadeClassifier("/home/csrobot/catkin_ws/src/spot_screwdriver/models/screwdriver_front_cascade.xml")

def angle_from_raw_image(img: cv2.Mat) -> float:
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

    found = screwdriver_locator.detectMultiScale(img, minNeighbors=5, minSize=(50,50), maxSize=(250,250))
    print(f"Cascade applied. Found {len(found)}")

    #if valid, focus in on screwdriver
    if len(found) == 1:
        #resize to square for orientation classifier
        x, y, w, h = found[0]

        l = (w + h) // 2
        print(f"l: {l}, w: {w}, h: {h}")
        c = (x + (w // 2), y + (h // 2))
        x, y = (c[0] - (l // 2), c[1] - (l // 2))

        img_cropped = img[y:y+l, x:x+l]
        img_cropped = cv2.flip(img_cropped, 1) #mirror
        img_median = int(np.median(img_cropped))
        print(f"Median: {img_median}")
        print(f"Mean: {np.mean(img_cropped)}")
        r, m1 = cv2.threshold(img_cropped, img_median, 255, cv2.THRESH_BINARY)
        h,w = img_cropped.shape
        r, m2 = cv2.threshold(img_cropped, max([img_cropped[0][0], img_cropped[0][w-1], img_cropped[h-1][0], img_cropped[h-1][w-1]]), 255, cv2.THRESH_BINARY_INV)
        m = m1 | m2
        cv2.imshow("Cropped", img_cropped)
        cv2.imshow("Mask", m)
        cv2.waitKey(0)
        #pass through classifier and return value
        return angle_classifer.get_orientation(img_cropped)
    #else return dummy value
    else:
        return 4.0

img = cv2.imread("/home/csrobot/catkin_ws/src/spot_screwdriver/test_orientation_srv/data/img6.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
angle = angle_from_raw_image(img)
print(angle)
