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

global iOCT_frame

MAX_FRAME = 100
DEBUG = False


def convert_ros_to_numpy(image_message):
    global iOCT_frame

    frame = image_to_numpy(image_message)
    iOCT_frame = cv2.resize(frame, (640, 480))


def try_publish(publishers, data, spec_strs=None):
    if len(publishers) != len(data):
        print(
            "Mismatch between number of publishers and data instances. Skipping publishing."
        )
        return

    if not spec_strs:
        spec_strs = [None] * len(publishers)

    for publisher, data_instance, spec_str in zip(publishers, data, spec_strs):
        if publisher:
            try:
                publisher.publish(data_instance)
                if DEBUG and spec_str:
                    print(f"Successfully published to {spec_str}: {data_instance}")
            except Exception as e:
                if DEBUG:
                    print(
                        f"Failed to publish data {data_instance} at publisher-{spec_str}: {e}"
                    )
                pass
        else:
            if DEBUG:
                print(
                    f"Publisher-{spec_str} is not defined, skipping publishing. Expected to publish {data_instance}"
                )


if __name__ == "__main__":

    # get publishers and subscribers
    numeric_publisher_spec = [
        "seg_posx_publisher",
        "seg_posy_publisher",
        "kal_posx_publisher",
        "kal_posy_publisher",
        "kal_velx_publisher",
        "kal_vely_publisher",
    ]
    numeric_publisher_arr = [
        # seg_posx_publisher
        rospy.Publisher("/image_model/SegmentPosX", Int32, queue_size=1),
        # seg_posy_publisher
        rospy.Publisher("/image_model/SegmentPosY", Int32, queue_size=1),
        # kal_posx_publisher
        rospy.Publisher("/image_model/KalmanPosX", Float64, queue_size=1),
        # kal_posy_publisher
        rospy.Publisher("/image_model/KalmanPosY", Float64, queue_size=1),
        # kal_velx_publisher
        rospy.Publisher("/image_model/KalmanVelX", Float64, queue_size=1),
        # kal_vely_publisher
        rospy.Publisher("/image_model/KalmanVelY", Float64, queue_size=1),
    ]
    mask_publisher = rospy.Publisher("/image_model/MaskImage", Image, queue_size=1)
    puncture_flag_publisher = rospy.Publisher(
        "/PunctureFlagImage",
        # Bool,
        Int32,
        queue_size=1,
    )
    model_publisher = rospy.Publisher("/image_model/ModelStartFlag", Bool, queue_size=1)
    iOCT_camera_subscriber = rospy.Subscriber(
        "/decklink/camera/image_raw", Image, convert_ros_to_numpy
    )
    # TODO get model path
    segmentation_model_path = "model_weights/unet-2.3k-augmented-wbase-wspaceaug.pth"
    image_processor = ImageProcessor(model_path=segmentation_model_path)

    rospy.init_node("image_puncture_detection", anonymous=True)
    time.sleep(0.5)

    def shutdown_event():
        print("ROS is shutting down. Publishing False.")
        model_publisher.publish(False)

    rospy.on_shutdown(shutdown_event)  # Register early!

    try:
        while not rospy.is_shutdown():
            numeric_data, mask, flag = image_processor.serialized_processing(iOCT_frame)
            try_publish(numeric_publisher_arr, numeric_data, numeric_publisher_spec)
            mask_msg = numpy_to_image(mask, encoding="mono8")
            try_publish([mask_publisher], [mask_msg])
            try_publish([puncture_flag_publisher], [flag], ["puncture_flag_publisher"])
            model_publisher.publish(True)

        rospy.spin()

    except KeyboardInterrupt:
        print("keyboard interrupt. Ending")

    finally:
        print("entered finally 2")
