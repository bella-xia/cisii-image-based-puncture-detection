#!/usr/bin/env python
import numpy as np
import cv2
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R

# from pytransform3d.trajectories import plot_trajectory
import time
import math

# for ros stuff
import rospy
import message_filters
from std_msgs.msg import String, Float64MultiArray, Bool, Float64, Float32, Time
from geometry_msgs.msg import Vector3, Transform
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import pandas as pd

# for dynamic reconfiguration of variables
import dynamic_reconfigure.client

# init variable
bridge = CvBridge()
isPunctureProbability = None
max_corr_values = None


class ros_topics:
    def __init__(self):
        self.bridge = CvBridge()

        # subscribers
        self.transform_sub = rospy.Subscriber(
            "/eye_robot/FrameEE", Transform, self.callback_2
        )
        # self.usb_camera_sub = rospy.Subscriber("/camera/image", Image, self.get_camera_image)
        self.iOCT_camera_sub = rospy.Subscriber(
            "/decklink/camera/image_raw", Image, self.get_iOCT_camera_image
        )
        self.b_scan_sub = rospy.Subscriber("/b_scan", Image, self.get_b_scan_image)
        self.cropped_image_sub = rospy.Subscriber(
            "/CroppedImage", Image, self.get_cropped_image
        )
        self.F_tip_sub = rospy.Subscriber(
            "/eye_robot/TipForceNormGlobal", Float64, self.get_F_tip
        )
        self.F_tip_EMA_sub = rospy.Subscriber(
            "/eye_robot/TipForceNormEMA_Global", Float64, self.get_F_tip_EMA
        )
        self.F_tip_AEWMA_sub = rospy.Subscriber(
            "/eye_robot/TipForceNormAEWMA_Global", Float64, self.get_F_tip_AEWMA
        )
        self.F_tip_dot_sub = rospy.Subscriber(
            "/eye_robot/TipForceNormDotGlobal", Float64, self.get_F_tip_dot
        )
        self.F_tip_dot_EMA_sub = rospy.Subscriber(
            "/eye_robot/TipForceNormDotEMA_Global", Float64, self.get_F_tip_dot_EMA
        )
        self.F_tip_dot_AEWMA_sub = rospy.Subscriber(
            "/eye_robot/TipForceNormDotAEWMA_Global", Float64, self.get_F_tip_dot_AEWMA
        )
        self.velocity_z_sub = rospy.Subscriber(
            "/eye_robot/robotVelZ_Global", Float64, self.get_velocity_z
        )
        self.puncture_prob_image_sub = rospy.Subscriber(
            "/PunctureProbabilityImage", Float32, self.get_puncture_prob_image
        )
        self.time_puncture_sub = rospy.Subscriber(
            "/eye_robot/TimePuncture_Global", Float64, self.get_time_puncture
        )
        self.Puncture_detection_flag_sub = rospy.Subscriber(
            "/eye_robot/PunctureDetectionFlag_Global", Int32, self.get_puncture_flag
        )
        self.Puncture_flag_image_sub = rospy.Subscriber(
            "/PunctureFlagImage", Bool, self.get_puncture_flag_image
        )

        time.sleep(1)

        self.vec3_x = None
        self.vec3_y = None
        self.vec3_z = None
        self.rot_x = None
        self.rot_y = None
        self.rot_ziOCT_image = None
        self.iOCT_image = None
        self.cropped_image = None
        self.b_scan = None
        self.F_tip = None
        self.F_tip_EMA = None
        self.F_tip_AEWMA = None
        self.F_tip_dot = None
        self.F_tip_dot_EMA = None
        self.F_tip_dot_AEWMA = None
        self.puncture_flag = None
        self.puncture_flag_image = None
        self.velocity_z = None
        self.time_puncture = None
        self.puncture_prob_image = None

        # Bella Hanbei

        # Synchronize the topics
        # self.ts = message_filters.TimeSynchronizer([self.iOCT_camera_sub, self.time_puncture_sub], 10)
        # self.ts.registerCallback(callback)

    def callback_2(self, data):
        self.vec3_x = data.translation.x
        self.vec3_y = data.translation.y
        self.vec3_z = data.translation.z
        self.rot_x = data.rotation.x
        self.rot_y = data.rotation.y
        self.rot_z = data.rotation.z
        self.rot_w = data.rotation.w

    # def get_camera_image(self,data):
    #   global usb_image
    #   usb_image = data

    def get_iOCT_camera_image(self, data):
        self.iOCT_image = data

    def get_cropped_image(self, data):
        self.cropped_image = data

    def get_b_scan_image(self, data):
        self.b_scan = data

    def get_F_tip(self, data):

        self.F_tip = data.data

    def get_F_tip_EMA(self, data):

        self.F_tip_EMA = data.data

    def get_F_tip_AEWMA(self, data):

        self.F_tip_AEWMA = data.data

    def get_F_tip_dot(self, data):

        self.F_tip_dot = data.data

    def get_F_tip_dot_EMA(self, data):
        self.F_tip_dot_EMA = data.data

    def get_F_tip_dot_AEWMA(self, data):
        self.F_tip_dot_AEWMA = data.data

    def get_puncture_flag(self, data):
        self.puncture_flag = data.data

    def get_puncture_prob_image(self, data):
        self.puncture_prob_image = data.data

    def get_puncture_flag_image(self, data):
        self.puncture_flag_image = data.data

    def get_velocity_z(self, data):
        self.velocity_z = data.data

    def get_time_puncture(self, data):
        self.time_puncture = data.data
        # Extract timestamp from the Time message
        # timeStamp = time_puncture.to_sec()


# Create ROS publishers and subscribers
rt = ros_topics()
rospy.init_node("rostopic_recorder", anonymous=True)
time.sleep(0.5)
num_frames = 0
ee_points = []

# create a new dir
time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

ep_dir = os.path.join("_recordings", time_stamp)
os.makedirs(ep_dir)

rate = rospy.Rate(30)  # ROS Rate at 5Hz

while not rospy.is_shutdown():
    # Capture frame-by-frame
    print("No. ", num_frames)
    iOCT_frame = bridge.imgmsg_to_cv2(rt.iOCT_image, desired_encoding="bgr8")
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))
    cv2.imshow("iOCT_frame", iOCT_frame)

    # save frame
    # iOCT_save_name = os.path.join(ep_dir, "iOCT_image_{:06d}".format(num_frames) + ".jpg")
    iOCT_save_name = os.path.join(
        ep_dir, "iOCT_image_{:.10f}.jpg".format(rt.time_puncture)
    )
    # write the image
    cv2.imwrite(
        iOCT_save_name, iOCT_frame
    )  # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)

    # save cropped image
    cropped_image_frame = bridge.imgmsg_to_cv2(
        rt.cropped_image, desired_encoding="bgr8"
    )
    cropped_image_frame = cv2.resize(cropped_image_frame, (640, 480))
    # cv2.imshow('cropped_image_frame', cropped_image_frame)

    # save frame
    # iOCT_save_name = os.path.join(ep_dir, "iOCT_image_{:06d}".format(num_frames) + ".jpg")
    cropped_image_save_name = os.path.join(
        ep_dir, "cropped_image_{:.10f}.jpg".format(rt.time_puncture)
    )
    # write the image
    cv2.imwrite(
        cropped_image_save_name, cropped_image_frame
    )  # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)

    # save b_scan
    # b_scan_frame = bridge.imgmsg_to_cv2(rt.b_scan, desired_encoding = 'passthrough')
    # b_scan_frame = cv2.resize(b_scan_frame, (640, 480))
    # cv2.imshow('b_scan_frame', b_scan_frame)

    # save frame
    # b_scan_save_name = os.path.join(ep_dir, "b_scan_{:06d}".format(num_frames) + ".jpg")
    # b_scan_save_name = os.path.join(ep_dir, "b_scan_{}.jpg".format(rt.time_puncture))

    # # write the image
    # cv2.imwrite(b_scan_save_name, b_scan_frame) # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)

    num_frames = num_frames + 1

    # get ee point
    ee_points.append(
        [
            rt.vec3_x,
            rt.vec3_y,
            rt.vec3_z,
            rt.rot_x,
            rt.rot_y,
            rt.rot_z,
            rt.rot_w,
            rt.velocity_z,
            rt.time_puncture,
            rt.F_tip,
            rt.F_tip_EMA,
            rt.F_tip_AEWMA,
            rt.F_tip_dot,
            rt.F_tip_dot_EMA,
            rt.F_tip_dot_AEWMA,
            rt.puncture_flag,
            rt.puncture_flag_image,
            rt.puncture_prob_image,
        ]
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # make sure we spin at 30hz
    rate.sleep()


# save ee points
header = [
    "vec3_x",
    "vec3_y",
    "vec3_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
    "velocity_z",
    "time_puncture",
    "F_tip",
    "F_tip_EMA",
    "F_tip_AEWMA",
    "F_tip_dot",
    "F_tip_dot_EMA",
    "F_tip_dot_AEWMA",
    "puncture_flag",
    "puncture_flag_image",
    "puncture_prob_image",
]
csv_data = pd.DataFrame(ee_points)
ee_save_path = os.path.join(ep_dir, "ee_csv.csv")
csv_data.to_csv(ee_save_path, index=False, header=header)

# When everything done, destroy all windows
cv2.destroyAllWindows()
