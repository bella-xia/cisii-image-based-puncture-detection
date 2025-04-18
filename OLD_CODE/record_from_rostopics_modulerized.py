#!/usr/bin/env python

# python lib imports
import cv2, os, time
from datetime import datetime
from collections import namedtuple
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend (headless)
import matplotlib.pyplot as plt
import pandas as pd

# ros imports
import rospy
from std_msgs.msg import String, Float64MultiArray, Bool, Float64, Float32, Time
from geometry_msgs.msg import Vector3, Transform
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from _utils_rospy.subscriber_module import SubRosTopic, RosTopicSubscriber

# init variable
bridge = CvBridge()
isPunctureProbability = None
max_corr_values = None

DATA_ASSIGNMENT = {"b_scan", "iOCT_image", "mask_image"}


class ros_topics:
    def __init__(self):
        self.bridge = CvBridge()

        # subscribers
        self.transform_sub = rospy.Subscriber(
            "/eye_robot/FrameEE", Transform, self.main_callback
        )

        self.ros_subscriber = RosTopicSubscriber(
            [
                # RosTopic("/eye_robot/FrameEE", Transform, lambda _: None),
                SubRosTopic(
                    "/decklink/camera/image_raw", Image, "iOCT_camera_sub", "iOCT_image"
                ),
                SubRosTopic("/b_scan", Image, "b_scan_sub", "b_scan"),
                SubRosTopic(
                    "/eye_robot/TipForceNormGlobal", Float64, "F_tip_sub", "F_tip"
                ),
                SubRosTopic(
                    "/eye_robot/TipForceNormEMA_Global",
                    Float64,
                    "F_tip_EMA_sub",
                    "F_tip_EMA",
                ),
                SubRosTopic(
                    "/eye_robot/TipForceNormAEWMA_Global",
                    Float64,
                    "F_tip_AEWMA_sub",
                    "F_tip_AEWMA",
                ),
                SubRosTopic(
                    "/eye_robot/TipForceNormDotGlobal",
                    Float64,
                    "F_tip_dot_sub",
                    "F_tip_dot",
                ),
                SubRosTopic(
                    "/eye_robot/TipForceNormDotEMA_Global",
                    Float64,
                    "F_tip_dot_EMA_sub",
                    "F_tip_dot_EMA",
                ),
                SubRosTopic(
                    "/eye_robot/TipForceNormDotAEWMA_Global",
                    Float64,
                    "F_tip_dot_AEWMA_sub",
                    "F_tip_dot_AEWMA",
                ),
                SubRosTopic(
                    "/eye_robot/robotVelZ_Global",
                    Float64,
                    "velocity_z_sub",
                    "velocity_z",
                ),
                SubRosTopic(
                    "/eye_robot/TimePuncture_Global",
                    Float64,
                    "time_puncture_sub",
                    "time_puncture",
                ),
                SubRosTopic(
                    "/eye_robot/PunctureDetectionFlag_Global",
                    Int32,
                    "puncture_detection_flag_sub",
                    "puncture_flag",
                ),
                # Bella Hanbei
                SubRosTopic(
                    "/PunctureFlagImage",
                    # Bool,
                    Int32,
                    "puncture_image_flag_sub",
                    "puncture_image_flag",
                ),
                SubRosTopic(
                    "/image_model/MaskImage", Image, "mask_image_sub", "mask_image"
                ),
                SubRosTopic(
                    "/image_model/SegmentPosX",
                    Float64,
                    "segment_pos_x_sub",
                    "segment_pos_x",
                ),
                SubRosTopic(
                    "/image_model/SegmentPosY",
                    Float64,
                    "segment_pos_y_sub",
                    "segment_pos_y",
                ),
                SubRosTopic(
                    "/image_model/KalmanPosX",
                    Float64,
                    "kalman_pos_x_sub",
                    "kalman_pos_x",
                ),
                SubRosTopic(
                    "/image_model/KalmanPosY",
                    Float64,
                    "kalman_pos_y_sub",
                    "kalman_pos_y",
                ),
                SubRosTopic(
                    "/image_model/KalmanVelX",
                    Float64,
                    "kalman_vel_x_sub",
                    "kalman_vel_x",
                ),
                SubRosTopic(
                    "/image_model/KalmanVelY",
                    Float64,
                    "kalman_vel_y_sub",
                    "kalman_vel_y",
                ),
                SubRosTopic(
                    "/image_model/ModelStartFlag",
                    Bool,
                    "model_start_flag_sub",
                    "model_start_flag",
                ),
            ]
        )

        time.sleep(1)

    def main_callback(self, data):
        self.vec3_x = data.translation.x
        self.vec3_y = data.translation.y
        self.vec3_z = data.translation.z
        self.rot_x = data.rotation.x
        self.rot_y = data.rotation.y
        self.rot_z = data.rotation.z
        self.rot_w = data.rotation.w


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

while (not rospy.is_shutdown()) and rt.model_start_flag:
    # Capture frame-by-frame
    print("No. ", num_frames)
    iOCT_frame = bridge.imgmsg_to_cv2(
        rt.ros_subscriber.get_data("iOCT_image"), desired_encoding="bgr8"
    )
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))
    iOCT_save_name = os.path.join(
        ep_dir,
        "iOCT_image_{:.10f}.jpg".format(rt.ros_subscriber.get_data("time_puncture")),
    )

    # write the image
    # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)
    cv2.imwrite(iOCT_save_name, iOCT_frame)

    # save mask
    mask_image = bridge.imgmsg_to_cv2(
        rt.ros_subscriber.get_data("mask_image"), desired_encoding="mono8"
    )
    mask_image_frame = cv2.resize(mask_image, (640, 480))
    # cv2.imshow('cropped_image_frame', cropped_image_frame)

    # Bella & Hanbei
    # save frame
    mask_image_save_name = os.path.join(
        ep_dir,
        "mask_image_{:.10f}.png".format(rt.ros_subscriber.get_data("time_puncture")),
    )
    # write the image
    # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)
    plt.imshow(mask_image_frame, cmap="gray")
    plt.axis("off")
    plt.savefig(mask_image_save_name)
    plt.close()
    # cv2.imwrite(mask_image_save_name, mask_image_frame)

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
            rt.ros_subscriber.get("velocity_z"),
            rt.ros_subscriber.get("time_puncture"),
            rt.ros_subscriber.get("F_tip"),
            rt.ros_subscriber.get("F_tip_EMA"),
            rt.ros_subscriber.get("F_tip_AEWMA"),
            rt.ros_subscriber.get("F_tip_dot"),
            rt.ros_subscriber.get("F_tip_dot_EMA"),
            rt.ros_subscriber.get("F_tip_dot_AEWMA"),
            rt.ros_subscriber.get("puncture_flag"),
            # Bella & Hanbei
            rt.ros_subscriber.get("puncture_image_flag"),
            rt.ros_subscriber.get("segment_pos_x"),
            rt.ros_subscriber.get("segment_pos_y"),
            rt.ros_subscriber.get("kalman_pos_x"),
            rt.ros_subscriber.get("kalman_pos_y"),
            rt.ros_subscriber.get("kalman_vel_x"),
            rt.ros_subscriber.get("kalman_vel_y"),
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
    # Bella & Hanbei
    "puncture_image_flag",
    "segment_pos_x",
    "segment_pos_y",
    "kalman_pos_x",
    "kalman_pos_y",
    "kalman_vel_x",
    "kalman_vel_y",
]

csv_data = pd.DataFrame(ee_points)
ee_save_path = os.path.join(ep_dir, "ee_csv.csv")
csv_data.to_csv(ee_save_path, index=False, header=header)

# When everything done, destroy all windows
cv2.destroyAllWindows()
