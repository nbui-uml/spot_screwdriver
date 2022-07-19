#!/usr/bin/env python3

#Standard ROS imports
import rospy
import rospkg

#CV related imports
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
from screwdriver_orientation_classifier import OrientationClassifier

#Service related imports
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from spot_screwdriver.srv import ScrewdriverOrientationSrv, ScrewdriverOrientationSrvRequest, ScrewdriverOrientationSrvResponse
from sensor_msgs.msg import Image

#tf related imports
import tf
import tf2_ros
import geometry_msgs

ros_pack = rospkg.RosPack()
L_cam_topic = rospy.get_param("L_front_cam_topic", "camera/frontleft/camera/image")
R_cam_topic = rospy.get_param("R_front_came_topic", "camera/frontright/camera/image")

bridge = CvBridge()
angle_classifer = OrientationClassifier()
screwdriver_locator = cv2.CascadeClassifier(ros_pack.get_path("spot_screwdriver") + "/models/screwdriver_front_cascade.xml") #is it better to initialize this here or in the function where it's used?

tf_broadcaster = None


def publish_transform_from_angle(angle: float) -> None:  
    """
    Publishes the tf frame from the gripper link to the screwdriver head.
    @param angle: type float, angle of the the screwdriver head in degrees.
    """  
    #init tf object
    static_tf_stamped = geometry_msgs.msg.TransformStamped()
    static_tf_stamped.header.stamp = rospy.Time.now()
    static_tf_stamped.header.frame_id = rospy.get_param("gripper_link", "gripper_link") #find actual name of this frame
    static_tf_stamped.child_frame_id = "screwdriver"

    #fill in translation of screwdriver
    static_tf_stamped.transform.translation.x = 0.15
    static_tf_stamped.transform.translation.y = 0
    static_tf_stamped.transform.translation.z = 0

    #convert angle to quartenion
    quat = tf.transformations.quaternion_from_euler(
        angle, 0, 0)
    static_tf_stamped.transform.rotation.x = quat[0]
    static_tf_stamped.transform.rotation.y = quat[1]
    static_tf_stamped.transform.rotation.z = quat[2]
    static_tf_stamped.transform.rotation.w = quat[3]

    #publish
    tf_broadcaster.sendTransform(static_tf_stamped)


def angle_from_raw_image(img: cv2.Mat) -> float:
    """
    Calculates the angle from the camera data.
    @param img: type cv2.Mat, image data from camera
    @return: type float, calculated angle of screwdriver; returns 400.0 if unable to calculate angle.
    """
    #detect screwdriver
    height, width, channels = img.shape
    scale_factor = 800 / height
    img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)

    found = screwdriver_locator.detectMultiScale(img, minNeighbors=5, minSize=(50,50), maxSize=(250,250))
    print(f"Cascade applied. Found {len(found)}")

    #if valid, focus in on screwdriver
    if len(found) == 1:
        #resize to square for orientation classifier
        x, y, w, h = found[0]
        
        """
        cpy = img.copy()
        cv2.rectangle(cpy, (x,y), (x+w,y+h), (0,0,255))
        cv2.imshow("C", cpy)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        """

        l = (w + h) // 2
        print(f"l: {l}, w: {w}, h: {h}")
        c = (x + (w // 2), y + (h // 2))
        x, y = (c[0] - (l // 2), c[1] - (l // 2))
        img_cropped = img[y:y+l, x:x+l]
        #pass through classifier and return value
        return angle_classifer.get_orientation(img_cropped)
    #else return dummy value
    else:
        return 400.0


def imgmsg_to_cv2(img_msg):
    """
    Helper function because I can't get CvBridge.imgmsg_to_cv2 to work.
    """
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byte order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv
    

def screwdriver_orientationCB(req: ScrewdriverOrientationSrvRequest) -> ScrewdriverOrientationSrvResponse:
    """
    Callback for the service request.
    Precondition: The arm is positioned in front of the head where the screwdriver can be viewed head-on.
    Postcondition: A transform frame from the tip of the screwdriver to the gripper is created and published.
    @param req: type ScrewdriverOrientationSrvRequest from spot_screwdriver.srv
    @return: type ScrewdriverOrientationSrvResponse from spot_screwdriver.srv
    """
    angle = 400.0
    tries = 0
    if req.topic == "":
        cam_topic = L_cam_topic
    else:
        cam_topic = req.topic
    while angle == 400.0 and tries < 10: #while not a valid angle and still under 10 tries
        img = rospy.wait_for_message(cam_topic, Image, 5)

        try:
            cv_img = imgmsg_to_cv2(img)
        except:
            print("Failed to convert to CV2")

        cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

        #probably create a haar cascade to recognize screwdriver from image, then isolate that region as a square img to pass to the orientation classifier
        angle = angle_from_raw_image(cv_img)
        print(f"angle: {angle}")

    #construct response object
    response = ScrewdriverOrientationSrvResponse()
    #if service failed to get the orientation of screwdriver
    if angle == 400.0: 
        response.success = False
    #otherwise, publish the angle
    else:
        response.success = True
        publish_transform_from_angle(angle)
    response.angle = angle

    return response


def server() -> None:
    rospy.init_node("screwdriver_orientation_server")
    global tf_broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    srv = rospy.Service("screwdriver_orientation", ScrewdriverOrientationSrv, screwdriver_orientationCB)
    print("screwdriver_orientation service ready")
    rospy.spin()

if __name__ == "__main__":
    server()

