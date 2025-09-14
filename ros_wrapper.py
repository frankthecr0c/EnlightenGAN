import numpy as np
import torch
import rospy
import os
import time
from pathlib import Path
import argparse  # Aggiunto
import cv2       # Aggiunto

from options.test_options import TestOptions
from models.models import create_model
from data.base_dataset import get_transform
from data.image_folder import store_dataset
from sensor_msgs.msg import Image, CompressedImage  # Modificato
from cv_bridge import CvBridge, CvBridgeError
from util.util import yaml_parser
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
    # --- Modificato: Aggiunto 'compressed' al costruttore ---
    def __init__(self, engan_opt, opt_ros, compressed=False, debug=False):
        self.debug = debug
        self.compressed = compressed  # Aggiunto
        self.error_flag = 0
        self.ros_opt = opt_ros
        self.encoding = self.ros_opt["Image"]["format"]
        self.EnGan_opt = engan_opt
        self.EnGan = create_model(engan_opt)
        self.bridge = CvBridge()

        # Set transform (see unaligned_dataset.py)
        self.transform = get_transform(engan_opt)
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

        # --- Modificato: Publisher/Subscriber condizionali ---
        # Definisce publisher/subscriber in base al flag 'compressed'
        if self.compressed:
            # Aggiunge il suffisso '/compressed' ai topic per usare il trasporto compresso
            in_topic = self.ros_opt["Node"]["Topics"]["enhancing_in"] + "/compressed"
            out_topic = self.ros_opt["Node"]["Topics"]["enhancing_out"] + "/compressed"
            self.image_pub = rospy.Publisher(out_topic, CompressedImage, queue_size=1)
            self.image_sub = rospy.Subscriber(in_topic, CompressedImage, self._img_callback, queue_size=1)
            rospy.loginfo(f"Utilizzo del trasporto di immagini COMPRESSE. In ascolto su: {in_topic}")
        else:
            in_topic = self.ros_opt["Node"]["Topics"]["enhancing_in"]
            out_topic = self.ros_opt["Node"]["Topics"]["enhancing_out"]
            self.image_pub = rospy.Publisher(out_topic, Image, queue_size=1)
            self.image_sub = rospy.Subscriber(in_topic, Image, self._img_callback, queue_size=1)
            rospy.loginfo(f"Utilizzo del trasporto di immagini RAW. In ascolto su: {in_topic}")


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
        #rospy.loginfo(msg)

        # --- Modificato: Gestione condizionale del messaggio in ingresso ---
        try:
            if self.compressed:
                # Converte CompressedImage in immagine cv2
                np_arr = np.frombuffer(img_msg.data, np.uint8)
                cv_image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                # La pipeline si aspetta un'immagine RGB, quindi convertiamo da BGR (default di OpenCV)
                cv_image_in = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
            else:
                # Converte Image (raw) in immagine cv2
                cv_image_in = self.bridge.imgmsg_to_cv2(img_msg, self.encoding)
        except (CvBridgeError, cv2.error) as e:
            msg = "Error while trying to convert ROS image to OpenCV: {}".format(e)
            rospy.logerr(msg)
            self.error_flag += 1
            return # Interrompe l'esecuzione se la conversione fallisce

        try:
            data = self._fake_unaligned_dataset_loader(cv_image_in)
        except Exception as e:
            msg = "Error while trying to prepare the fake data input for feeding the network: {}".format(e)
            rospy.logerr(msg)
            self.error_flag += 10
            return

        try:
            image_numpy, time_forward = self._engan_process(data)
        except Exception as e:
            msg = "Error while feeding and forwarding the network: {}".format(e)
            rospy.logerr(msg)
            print(msg)
            self.error_flag += 15
            return

        msg = "Publishing Back the image on the topic: {}".format(self.image_pub.name)
        #rospy.loginfo(msg)

        # --- Modificato: Gestione condizionale del messaggio in uscita ---
        try:
            if self.compressed:
                # L'output della rete 'image_numpy' è in formato RGB.
                # Convertilo in BGR prima di comprimerlo in JPEG.
                image_numpy_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                ros_image_out = self.bridge.cv2_to_compressed_imgmsg(image_numpy_bgr, dst_format='jpg')
            else:
                # Invia l'immagine raw, assumendo che 'image_numpy' sia RGB e self.encoding sia 'rgb8'
                ros_image_out = self.bridge.cv2_to_imgmsg(image_numpy, encoding=self.encoding)
        except CvBridgeError as e:
            msg = "Error while trying to convert OpenCV image to ROS: {}".format(e)
            rospy.logerr(msg)
            self.error_flag += 30
            return

        try:
            self.image_pub.publish(ros_image_out)
        except Exception as e:
            msg = "Error while trying to publish the enhanced image: {}".format(e)
            rospy.logerr(msg)
            print(msg)
            self.error_flag += 60
            return

        if self.error_flag == 0:
            msg = "DONE!,  FORWARD FPS = {}\n".format(1 / time_forward) + "-" * 50
        else:
            msg = "\nErrors occurred during processing the image\n\t -> error code: {}".format(
                self.error_flag) + "-" * 50

        #rospy.loginfo(msg)


if __name__ == "__main__":
   
 # --- Aggiunto: Argparse per gestire gli argomenti da console ---
#    parser = argparse.ArgumentParser(description="EnlightenGAN ROS Node.")
#    parser.add_argument('--compressed', action='store_true',
#                        help='Use compressed image transport for input and output topics.')
    # Usa parse_known_args() per ignorare gli argomenti specifici di ROS
#    args, unknown = parser.parse_known_args()

    # Ros node initialization
    rospy.init_node("EnlightenGAN_node", anonymous=False)

    # get and set option args
    eng_opt = TestOptions().parse()
    eng_opt.nThreads = 0   # test code only supports nThreads = 1
    eng_opt.batchSize = 1  # test code only supports batchSize = 1
    eng_opt.serial_batches = True  # no shuffle
    eng_opt.no_flip = True  # no flip

    # Get the ros configs, assuming they are in the config folder which is in the same level of this script
    script_path = Path.cwd()
    config_path = Path(script_path, "configs", "ros_config.yaml")
    ros_opt = yaml_parser(config_path)

    # --- Modificato: Crea l'handler passando l'argomento 'compressed' ---
    handler = RosEnGan(engan_opt=eng_opt, opt_ros=ros_opt, compressed=eng_opt.compressed)

    # Start ros loop
    rospy.spin()
