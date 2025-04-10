import time, cv2
from image_based_util_unet import ImageProcessor

# ROS imports
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float64, Int32
from image_conversion_without_using_ros import image_to_numpy

global iOCT_frame


def convert_ros_to_numpy(image_message):
    global iOCT_frame

    iOCT_frame = image_to_numpy(image_message)
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))


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
                print(publisher)
                publisher.publish(data_instance)
                if spec_str:
                    print(
                        f"Successfully published to {spec_str}. Expected to publish {data_instance}"
                    )
            except Exception as e:
                print(
                    f"Failed to publish data {data_instance} at publisher-{spec_str}: {e}"
                )
                pass
        else:
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
        "/image_model/PunctureImageFlag", Bool, queue_size=1
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
    model_publisher.publish(True)
    try:
        while not rospy.is_shutdown():
            numeric_data, mask, flag = image_processor.serialized_processing(iOCT_frame)
            try_publish(numeric_publisher_arr, numeric_data, numeric_publisher_spec)
            try_publish([mask_publisher], [mask])
            try_publish([puncture_flag_publisher], [flag], ["puncture_flag_publisher"])
    except KeyboardInterrupt:
        model_publisher.publish(False)
        print("Keyboard interrupted. Shutting down...")
        exit(0)

    model_publisher.publish(False)
