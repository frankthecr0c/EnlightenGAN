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

    img = cv2.imread(image_path)

    if img is None:
        rospy.logerr("Failed to load image from: {}".format(image_path))
        return

    ros_image = bridge.cv2_to_imgmsg(img, "bgr8")  # Assuming BGR image
    rospy.loginfo("Publishing image to topic: {}".format(topic_name))
    image_pub.publish(ros_image)
    rospy.sleep(1)  # Give some time for the message to be sent


if __name__ == '__main__':
    rospy.init_node('image_publisher')
    image_path = '/root/Archive/Shared/00010.jpg'  # Replace with your actual image path
    topic_name = 'img_req_enhancer'  # Replace with your desired topic name

    publish_image(image_path, topic_name)