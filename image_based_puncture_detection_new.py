import time, cv2, logging
from image_based_util_unet import ImageProcessor

# ROS imports
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float64, Int32
from image_conversion_without_using_ros import image_to_numpy

global iOCT_frame

logger = logging.getLogger(__name__)
logging.basicConfig(
    # filename="logger.out",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def convert_ros_to_numpy(image_message):
    global iOCT_frame

    iOCT_frame = image_to_numpy(image_message)
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))
    logger.info(iOCT_frame.shape)


if __name__ == "__main__":

    # get publishers and subscribers
    seg_posx_publisher = rospy.Publisher(
        "/image_model/SegmentPosX", Int32, queue_size=1
    )
    seg_posy_publisher = rospy.Publisher(
        "/image_model/SegmentPosY", Int32, queue_size=1
    )
    kal_posx_publisher = rospy.Publisher(
        "/image_model/KalmanPosX", Float64, queue_size=1
    )
    kal_posy_publisher = rospy.Publisher(
        "/image_model/KalmanPosY", Float64, queue_size=1
    )
    kal_velx_publisher = rospy.Publisher(
        "/image_model/KalmanVelX", Float64, queue_size=1
    )
    kal_vely_publisher = rospy.Publisher(
        "/image_model/KalmanVelY", Float64, queue_size=1
    )
    mask_publisher = rospy.Publisher("PunctureFlagImage", Image, queue_size=1)
    puncture_flag_publisher = rospy.Publisher(
        "/image_model/PunctureImageFlag", Bool, queue_size=1
    )
    iOCT_camera_subscriber = rospy.Subscriber(
        "/decklink/camera/image_raw", Image, convert_ros_to_numpy
    )

    # TODO get model path
    segmentation_model_path = "CISII/model_weights/my_model_weights.pth"
    image_processor = ImageProcessor(
        model_path=segmentation_model_path,
        seg_posx_publisher=seg_posx_publisher,
        seg_posy_publisher=seg_posy_publisher,
        kal_posx_publisher=kal_posx_publisher,
        kal_posy_publisher=kal_posy_publisher,
        kal_velx_publisher=kal_velx_publisher,
        kal_vely_publisher=kal_vely_publisher,
        mask_publisher=mask_publisher,
        puncture_flag_publisher=puncture_flag_publisher,
        logger=logger
    )

    rospy.init_node("image_puncture_detection", anonymous=True)
    time.sleep(0.5)
    while not rospy.is_shutdown():
        seg_posx, seg_posy, kal_posx, kal_posy, kal_velx, kal_vely, mask, flag = image_processor.serialized_processing(iOCT_frame)
        seg_posx_publisher.publish(seg_posx)
        seg_posy_publisher.publish(seg_posy)
        kal_posx_publisher.publish(kal_posx)
        kal_posy_publisher.publish(kal_posy)
        kal_velx_publisher.publish(kal_velx)
        kal_vely_publisher.publish(kal_vely)
        mask_publisher.publish(mask)
        puncture_flag_publisher.publish(flag)