import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from util.util import yaml_parser
import yaml
import cv2
import sys
import os
from pathlib import Path


class EnhancerScheduler:
    def __init__(self, cfg):
        self.options = cfg
        self.topics_opt = self.options["Node"]["Topics"]
        self.debug_mode = bool(self.options["Debugging"])
        self.send_to_enhancing = rospy.Publisher(self.topics_opt["enhancing_in"], Image, queue_size=1)
        self.receive_from_camera = rospy.Subscriber(self.topics_opt["camera_out"], Image, self._img_in_callback)
        self._initialize_ros_timer()
        self.new_msg_available = False
        self.latest_img_msg = None

    def _initialize_ros_timer(self):
        rate_send_sec = (opt["Scheduler"]["time_msec"]) / 1000.0
        rospy.loginfo("Scheduler Initialized at {} Hz".format(1/rate_send_sec))
        self.timer = rospy.Timer(rospy.Duration(rate_send_sec), self._timer_callback)

    def _timer_callback(self, event):
        if self.new_msg_available:

            self.send_to_enhancing.publish(self.latest_img_msg)
            if self.debug_mode:
                rospy.loginfo("Incoming image detected: {} ---> {}"
                              .format(self.topics_opt["camera_out"], self.topics_opt["enhancing_in"]))
            self.new_msg_available = False
        else:
            if self.debug_mode:
                rospy.logwarn("No now image detected, check the publishing node: {}"
                              .format(self.topics_opt["enhancing_in"]))
            else:
                pass

    def _img_in_callback(self, img_msg):
        self.latest_img_msg = img_msg
        self.new_msg_available = True


if __name__ == '__main__':

    # Get the configs assuming the .yaml file is stored in the "../config" folder
    script_path = Path.cwd().parent
    config_path = Path(script_path, "configs", "ros_config.yaml")
    opt = yaml_parser(config_path)

    # Prepare node and subscriber/publisher
    rospy.init_node(opt["Node"]["Name"])

    # Create the class
    scheduler = EnhancerScheduler(cfg=opt)

    # Start the scheduler
    rospy.spin()
