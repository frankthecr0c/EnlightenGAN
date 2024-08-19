import numpy as np
import torch
import rospy
from options.test_options import TestOptions
from models.models import create_model
from data.base_dataset import get_transform
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torchvision.transforms as transforms
from PIL import Image as PImage


def display_image_pil(image_tensor):
    """Displays an image tensor using PIL (Pillow).

    Args:
        image_tensor: The image tensor to be displayed.
    """

    image_np = image_tensor.numpy()

    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)

    image_np = (image_np * 255).astype(np.uint8)

    image = PImage.fromarray(image_np)
    image.show()



class RosEnGan:
    def __init__(self, engan_opt):
        self.EnGan_opt = engan_opt
        self.EnGan = create_model(engan_opt)
        self.transform = get_transform(engan_opt)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/img_enhanced", Image, queue_size=1)
        self.image_sub = rospy.Subscriber("/img_req_enhancer", Image, self._img_callback)
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)



    def _img_callback(self, img_msg):
        rospy.loginfo("New image enhancement request!")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            A_img = PImage.fromarray(cv_image)
            A_img = self.transform(A_img)
            display_image_pil(A_img)

        except Exception as e:
            rospy.logerr(e)
            print("Error while trying to convert ROS image to OpenCV: {}".format(e))



if __name__ == "__main__":
    # Ros node initialization
    rospy.init_node("EnlightenGAN_node", anonymous=False)

    # get and set option args
    opt = TestOptions().parse()
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    handler = RosEnGan(engan_opt=opt)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()




# def _img_callback(self, img_msg):
    #     rospy.loginfo("New image enhancement request!")
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
    #
    #         # Ensure the image is in BGR format (OpenCV's default)
    #         A_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #
    #         # Get the original image size
    #         A_height, A_width = A_img.shape[:2]
    #
    #         # Calculate the new size, ensuring it's divisible by 16
    #         new_A_width = (A_width // 16) * 16
    #         new_A_height = (A_height // 16) * 16
    #
    #         # Resize the image using OpenCV
    #         A_img = cv2.resize(A_img, (new_A_width, new_A_height), interpolation=cv2.INTER_CUBIC)
    #
    #
    #         cv2.imshow("Original Image", A_img)
    #         cv2.waitKey(0)  # Wait until a key is pressed
    #         cv2.destroyAllWindows()  # Close all windows
    #
    #         # Convert to PyTorch tensor
    #         A_img_tensor = torch.from_numpy(A_img)
    #
    #         # Change the shape from (H, W, C) to (C, H, W)
    #         A_img_tensor = A_img_tensor.permute(2, 0, 1)
    #         A_grayscale_tensor = self.to_grayscale(A_img_tensor)
    #
    #         if self.EnGan_opt.gpu_ids:
    #             A_img_tensor = A_img_tensor.cuda().float()/ 255.0
    #             A_grayscale_tensor = A_grayscale_tensor.cuda().float()/ 255.0
    #
    #         self.EnGan.input_A = A_img_tensor.unsqueeze(0)
    #         # self.EnGan.input_B = A_img_tensor.unsqueeze(0)
    #         self.EnGan.input_A_gray = A_grayscale_tensor.unsqueeze(0)
    #
    #         # Convert back to numpy
    #         A_img_enh = self.EnGan.predict()["fake_B"]
    #
    #         # Convert from RGB to BGR (OpenCV's format)
    #         A_img_enh_cv = cv2.cvtColor(A_img_enh, cv2.COLOR_RGB2BGR)
    #
    #         cv2.imshow("Enhanced Image", A_img_enh_cv)
    #         cv2.waitKey(0)  # Wait until a key is pressed
    #         cv2.destroyAllWindows()  # Close all windows
    #
    #         # Convert to ros image message
    #         A_img_enh_ros = self.bridge.cv2_to_imgmsg(A_img_enh_cv, "bgr8")  # Assuming BGR image
    #         # Publish the image
    #         self.image_pub.publish(A_img_enh_ros)
    #
    #     except CvBridgeError as e:
    #         rospy.logerr(e)
    #         print("Error whiel trying to convert ROS image to OpenCV: {}".format(e))
    #
    #
    #     try:
    #         self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    #     except CvBridgeError as e:
    #         print(e)