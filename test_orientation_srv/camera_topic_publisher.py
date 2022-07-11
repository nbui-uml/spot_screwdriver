#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

L_cam_topic = rospy.get_param("L_front_cam_topic", "camera/frontleft/camera/image")
R_cam_topic = rospy.get_param("R_front_came_topic", "camera/frontright/camera/image")

bridge = CvBridge()


rospy.init_node("camera_topic_publisher")
L_pub = rospy.Publisher(L_cam_topic, Image, queue_size=10)
R_pub = rospy.Publisher(R_cam_topic, Image, queue_size=10)

L_img = cv2.imread("/home/csrobot/catkin_ws/src/spot_screwdriver/test_orientation_srv/img/L.jpg")
R_img = cv2.imread("/home/csrobot/catkin_ws/src/spot_screwdriver/test_orientation_srv/img/R.jpg")

L_img = cv2.resize(cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY), (0,0), fx=0.5, fy=0.5)
R_img = cv2.resize(cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY), (0,0), fx=0.5, fy=0.5)

L_img = cv2.cvtColor(L_img, cv2.COLOR_GRAY2BGR)
R_img = cv2.cvtColor(R_img, cv2.COLOR_GRAY2BGR)

try:
    L_msg = bridge.cv2_to_imgmsg(L_img)
    R_msg = bridge.cv2_to_imgmsg(R_img)
except CvBridgeError as e:
    print(e)

rate = rospy.Rate(4)

while not rospy.is_shutdown():
    L_msg.header.stamp.set(rospy.Time.now().secs, rospy.Time.now().nsecs)
    R_msg.header.stamp.set(rospy.Time.now().secs, rospy.Time.now().nsecs)
    L_pub.publish(L_msg)
    R_pub.publish(R_msg)
    rate.sleep()