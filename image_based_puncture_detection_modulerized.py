# python lib imports
import time, cv2

# ROS imports
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float64, Int32

# self-defined function imports
from _utils_model.image_based_util_unet import ImageProcessor
from _utils_image.image_conversion_without_using_ros import (
    numpy_to_image,
    image_to_numpy,
)
from _utils_rospy.publisher_module import PubRosTopic, RosTopicPublisher
from _utils_rospy.subscriber_module import SubRosTopic, RosTopicSubscriber


def convert_ros_to_numpy(image_message):
    global iOCT_frame

    frame = image_to_numpy(image_message)
    iOCT_frame = cv2.resize(frame, (640, 480))


IMAGE_MODEL_ROSTOPIC = [
    "/image_model/SegmentPosX",
    "/image_model/SegmentPosY",
    "/image_model/KalmanPosX",
    "/image_model/KalmanPosY",
    "/image_model/KalmanVelX",
    "/image_model/KalmanVelY",
]

IMAGE_MODEL_DATA_PUBLISHER = [
    "seg_posx_publisher",
    "seg_posy_publisher",
    "kal_posx_publisher",
    "kal_posy_publisher",
    "kal_velx_publisher",
    "kal_vely_publisher",
]

if __name__ == "__main__":

    publisher_manager = RosTopicPublisher(
        [
            PubRosTopic(
                IMAGE_MODEL_ROSTOPIC[idx], Float64, IMAGE_MODEL_DATA_PUBLISHER[idx]
            )
            for idx in range(len(IMAGE_MODEL_ROSTOPIC))
        ]
        + [
            PubRosTopic("/image_model/MaskImage", Image, "mask_publisher"),
            PubRosTopic("/PunctureFlagImage", Bool, "flag_publisher"),
            PubRosTopic("/image_model/ModelStartFlag", Bool, "starter_publisher"),
        ]
    )
    subscriber_manager = RosTopicSubscriber(
        [
            SubRosTopic(
                "/decklink/camera/image_raw",
                Image,
                "iOCT_camera_subscriber",
                "iOCT_frame",
            )
        ]
    )

    segmentation_model_path = "model_weights/unet-2.3k-augmented-wbase-wspaceaug.pth"
    image_processor = ImageProcessor(model_path=segmentation_model_path)

    rospy.init_node("image_puncture_detection", anonymous=True)
    time.sleep(0.5)

    def shutdown_event():
        print("ROS is shutting down. Publishing False.")
        publisher_manager.publish_data(["starter_publisher"], [False])

    rospy.on_shutdown(shutdown_event)

    try:
        while not rospy.is_shutdown():
            image = cv2.resize(subscriber_manager.get_data("iOCT_frame", (640, 480)))
            numeric_data, mask, flag = image_processor.serialized_processing(image)
            publisher_manager.publish_data(IMAGE_MODEL_DATA_PUBLISHER, numeric_data)
            publisher_manager.publish_data(
                ["mask_publisher"], [numpy_to_image(mask, encoding="mono8")]
            )
            publisher_manager.publish_data(["flag_publisher"], [flag])
            publisher_manager.publish_data(["starter_publisher"], [True])

        rospy.spin()

    except KeyboardInterrupt:
        print("keyboard interrupt. Ending")

    finally:
        print("Image-based Puncture Detection Exiting.... ")
