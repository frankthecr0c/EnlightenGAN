import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def img_callback(img_msg):
    bridge = CvBridge()
    cv_image_in = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
    cv2.imshow("Enhanced Image", cv_image_in)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close all windows


if __name__ == '__main__':
    rospy.init_node('test_receiver_enhanced')
    image_sub = rospy.Subscriber("/img_enhanced", Image, img_callback)
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
        rospy.spin()
