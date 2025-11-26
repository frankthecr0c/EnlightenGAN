import numpy as np
import torch
import rospy
import os
import time
from pathlib import Path
import argparse  # Aggiunto
import cv2       # Aggiunto
import threading

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
        
        # Parametri per l'immagine compressa
        self.compressed_format = self.ros_opt["Image"].get("compressed_format", "jpeg").lower()
        self.quality = self.ros_opt["Image"].get("quality", 90)
        self.show_output = self.ros_opt["Image"].get("show_output", False)
        
        # Crea finestre cv2 se necessario
        if self.show_output:
            cv2.namedWindow("EnlightenGAN Input (Original)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("EnlightenGAN Output (Enhanced)", cv2.WINDOW_NORMAL)
            rospy.loginfo("Finestre di visualizzazione cv2 abilitate (Input e Output)")
        
        # Usa un buffer per l'ultimo messaggio ricevuto + lock per thread-safety
        self.latest_msg = None
        self.msg_lock = threading.Lock()
        self.new_msg_available = False
        self.skipped_messages = 0
        
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
        
        # Avvia il thread di processing
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        rospy.loginfo("Thread di processing avviato - modalità drop-frame attiva")

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
        """Callback velocissimo: salva solo l'ultimo messaggio ricevuto"""
        with self.msg_lock:
            if self.new_msg_available:
                self.skipped_messages += 1
            self.latest_msg = img_msg
            self.new_msg_available = True
    
    def _processing_loop(self):
        """Loop separato che processa solo l'ultimo messaggio disponibile"""
        rospy.loginfo("Processing loop started")
        
        while not rospy.is_shutdown():
            # Controlla se c'è un nuovo messaggio da processare
            with self.msg_lock:
                if not self.new_msg_available:
                    msg_to_process = None
                else:
                    msg_to_process = self.latest_msg
                    self.new_msg_available = False
                    skipped = self.skipped_messages
                    self.skipped_messages = 0
            
            # Nessun messaggio da processare, aspetta un po'
            if msg_to_process is None:
                time.sleep(0.001)  # Sleep brevissimo per non consumare CPU
                continue
            
            if skipped > 0:
                rospy.loginfo(f"Processing latest image (skipped {skipped} frames)")
            
            self.error_flag = 0
            
            # --- Conversione messaggio ROS in cv2 ---
            try:
                if self.compressed:
                    # Converte CompressedImage in immagine cv2
                    np_arr = np.frombuffer(msg_to_process.data, np.uint8)
                    cv_image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    # La pipeline si aspetta un'immagine RGB
                    cv_image_in = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
                else:
                    # Converte Image (raw) in immagine cv2
                    cv_image_in = self.bridge.imgmsg_to_cv2(msg_to_process, self.encoding)
                
                # Visualizza l'immagine originale se abilitato
                if self.show_output:
                    if self.encoding == "rgb8" or not self.compressed:
                        cv_image_display = cv2.cvtColor(cv_image_in, cv2.COLOR_RGB2BGR)
                    else:
                        cv_image_display = cv_image_bgr if self.compressed else cv_image_in
                    cv2.imshow("EnlightenGAN Input (Original)", cv_image_display)
                    cv2.waitKey(1)
                    
            except (CvBridgeError, cv2.error) as e:
                rospy.logerr(f"Error converting ROS image to OpenCV: {e}")
                continue
            
            # --- Preparazione dati per la rete ---
            try:
                data = self._fake_unaligned_dataset_loader(cv_image_in)
            except Exception as e:
                rospy.logerr(f"Error preparing data for network: {e}")
                continue
            
            # --- Processing con EnlightenGAN ---
            try:
                image_numpy, time_forward = self._engan_process(data)
            except Exception as e:
                rospy.logerr(f"Error during network forward: {e}")
                continue
            
            # Visualizza l'immagine processata se abilitato
            if self.show_output:
                image_display = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                cv2.imshow("EnlightenGAN Output (Enhanced)", image_display)
                cv2.waitKey(1)
            
            # --- Conversione output in messaggio ROS ---
            try:
                if self.compressed:
                    image_numpy_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                    
                    if self.compressed_format == "jpeg" or self.compressed_format == "jpg":
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                        _, compressed_data = cv2.imencode('.jpg', image_numpy_bgr, encode_param)
                        ros_image_out = CompressedImage()
                        ros_image_out.header.stamp = rospy.Time.now()
                        ros_image_out.format = "jpeg"
                        ros_image_out.data = compressed_data.tobytes()
                    elif self.compressed_format == "png":
                        compression_level = 9 - min(9, max(0, int(self.quality / 11)))
                        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level]
                        _, compressed_data = cv2.imencode('.png', image_numpy_bgr, encode_param)
                        ros_image_out = CompressedImage()
                        ros_image_out.header.stamp = rospy.Time.now()
                        ros_image_out.format = "png"
                        ros_image_out.data = compressed_data.tobytes()
                    else:
                        rospy.logwarn(f"Formato '{self.compressed_format}' non riconosciuto, uso JPEG")
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                        _, compressed_data = cv2.imencode('.jpg', image_numpy_bgr, encode_param)
                        ros_image_out = CompressedImage()
                        ros_image_out.header.stamp = rospy.Time.now()
                        ros_image_out.format = "jpeg"
                        ros_image_out.data = compressed_data.tobytes()
                else:
                    ros_image_out = self.bridge.cv2_to_imgmsg(image_numpy, encoding=self.encoding)
            except CvBridgeError as e:
                rospy.logerr(f"Error converting OpenCV image to ROS: {e}")
                continue
            
            # --- Pubblicazione ---
            try:
                self.image_pub.publish(ros_image_out)
                rospy.loginfo(f"Published enhanced image - FPS: {1/time_forward:.2f}")
            except Exception as e:
                rospy.logerr(f"Error publishing enhanced image: {e}")
                continue


if __name__ == "__main__":
    
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

    # Verifica se l'opzione compressed è presente in eng_opt, altrimenti usa False come default
    use_compressed = getattr(eng_opt, 'compressed', False)
    
    # Crea l'handler passando l'argomento 'compressed'
    handler = RosEnGan(engan_opt=eng_opt, opt_ros=ros_opt, compressed=use_compressed)

    # Start ros loop
    rospy.spin()
