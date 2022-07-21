import cv2
import numpy as np

img = cv2.imread("/home/csrobot/catkin_ws/src/spot_screwdriver/test_screwdriver_detection/img/illuminated.jpg")
img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
img = cv2.GaussianBlur(img, (5,5), 1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = np.array([0,0,100])
upper_range = np.array([255,50,255])

mask = cv2.inRange(hsv, lower_range, upper_range)
mask = cv2.bitwise_not(mask)

circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=100, maxRadius=500)

clip = None

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
print(i, j)

cv2.imshow("out", img)
cv2.imshow("clip", clip)
cv2.waitKey(0)
cv2.destroyAllWindows()