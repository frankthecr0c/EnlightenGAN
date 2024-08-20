import copy
import numpy as np
import torch
import rospy
import os
import cv2
import time
from copy import deepcopy
from options.test_options import TestOptions
from models.models import create_model
from data.base_dataset import get_transform
from data.image_folder import store_dataset
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import torchvision.transforms as transforms
from PIL import Image as PImage


def display_image_pil(image_tensor):
    """Displays an image tensor using PIL (Pillow).

    Args:
        image_tensor: The image tensor to be displayed.
    """
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.numpy()
    else:
        image_np = image_tensor
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)

    image_np = (image_np * 255).astype(np.uint8)

    image = PImage.fromarray(image_np)
    image.show()


class RosEnGan:
    def __init__(self, engan_opt, encoding="rgb8", debug=False):
        self.debug = debug
        self.error_flag = 0
        self.encoding = encoding
        self.EnGan_opt = engan_opt
        self.EnGan = create_model(engan_opt)
        self.bridge = CvBridge()
        # Set transform (see unaligned_dataset.py)
        self.transform = get_transform(engan_opt)

        self.image_pub = rospy.Publisher("/img_enhanced", Image, queue_size=1)
        self.image_sub = rospy.Subscriber("/img_req_enhancer", Image, self._img_callback)
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

        # At least one B image is necessary for running (see unaligned_dataset.py)
        self.dir_B = os.path.join(self.EnGan_opt.dataroot, self.EnGan_opt.phase + 'B')
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)
        self.B_size = len(self.B_paths)
        # -> Get the first one index = 0
        self.B_img = self.B_imgs[0 % self.B_size]
        self.B_path = self.B_paths[0 % self.B_size]
        self.B_img = self.transform(self.B_img)


    def _fake_unaligned_dataset_loader(self, cv_image):

        # Create A image
        A_img = PImage.fromarray(cv_image)
        A_img = self.transform(A_img)

        # Create A_gray image
        r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.

        # Unsqueeze
        A_img = torch.unsqueeze(A_img, 0)
        input_img = A_img
        A_gray = torch.unsqueeze(A_gray, 0)
        A_gray = A_gray.unsqueeze(0)
        self.B_img = torch.unsqueeze(self.B_img, 0)

        if self.debug:
            display_image_pil(A_gray)
            display_image_pil(self.B_img)
            display_image_pil(A_img)
        # A_gray = (1./A_gray)/255.

        data = {'A': A_img, 'B': self.B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': "not_required", 'B_paths': self.B_path}

        return data

    def _engan_process(self, data):

        # Set input for the model
        self.EnGan.set_input(data)


        # Get the enhanced image, forwarding
        star_t = time.time()
        visuals = self.EnGan.predict()
        image_numpy = visuals["fake_B"].squeeze()
        avg_time = time.time() - star_t

        # If debug true, show the output image
        if self.debug:
            image = PImage.fromarray(image_numpy, 'RGB')
            image.show()

        return image_numpy, avg_time

    def _img_callback(self, img_msg):

        msg = "\nNew image enhancement request!"
        self.error_flag = 0
        rospy.loginfo(msg)

        try:
            cv_image_in = self.bridge.imgmsg_to_cv2(img_msg, self.encoding)
        except CvBridgeError as e:
            msg = "Error while trying to convert ROS image to OpenCV: {}".format(e)
            rospy.logerr(msg)
            self.error_flag += 1

        try:
            data = self._fake_unaligned_dataset_loader(cv_image_in)
        except Exception as e:
            msg = "Error while trying to prepare the fake data input for feeding the network: {}".format(e)
            rospy.logerr(msg)
            self.error_flag += 10

        try:
            image_numpy, time_forward = self._engan_process(data)
        except Exception as e:
            msg = "Error while feeding and forwarding the network: {}".format(e)
            rospy.logerr(msg)
            print(msg)
            self.error_flag += 15

        msg = "Publishing Back the image on the topic: {}".format(self.image_pub.name)
        rospy.loginfo(msg)

        try:
            ros_image_out = self.bridge.cv2_to_imgmsg(image_numpy, encoding=self.encoding)
        except CvBridgeError as e:
            msg = "Error while trying to convert OpenCV image to ROS: {}".format(e)
            rospy.logerr(msg)
            self.error_flag += 30

        try:
            self.image_pub.publish(ros_image_out)
        except Exception as e:
            msg = "Error while trying to publish the enhanced image: {}".format(e)
            rospy.logerr(msg)
            print(msg)
            self.error_flag += 60

        if self.error_flag == 0:
            msg = "DONE!,  FORWARD FPS = {}\n".format(1/time_forward) + "-" * 50
        else:
            msg = "\nErrors occurred during processing the image\n\t -> error code: {}".format(self.error_flag) + "-" * 50

        rospy.loginfo(msg)


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
