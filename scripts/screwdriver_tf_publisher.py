import rospy
import tf
import tf2_ros
import geometry_msgs

def publish(angle):    
   #init
   rospy.init_node("Screwdriver_tf_publisher")
   tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

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

   #publish
   tf_broadcaster.sendTransform(static_tf_stamped)
   rospy.spin()

