import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def publish_image(image_path, topic_name):
    """
    Publishes an image from a file path to a specified ROS topic.

    Args:
        image_path (str): The path to the image file.
        topic_name (str): The name of the ROS topic to publish to.
    """

    image_pub = rospy.Publisher(topic_name, Image, queue_size=10)
    bridge = CvBridge()

    # Display image
    img = cv2.imread(image_path)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)  # Wait until a key is pressed
    #cv2.destroyAllWindows()  # Close all windows

    if img is None:
        rospy.logerr("Failed to load image from: {}".format(image_path))
        return

    ros_image = bridge.cv2_to_imgmsg(img, "bgr8")  # Assuming BGR image
    rospy.loginfo("Publishing image to topic: {}".format(topic_name))
    image_pub.publish(ros_image)


if __name__ == '__main__':
    rospy.init_node('image_publisher')
    image_path = '/root/Archive/Shared/00010.jpg'   # test image path
    topic_name = 'img_req_enhancer'                 # desired publish topic name

    publish_image(image_path, topic_name)           # publish
