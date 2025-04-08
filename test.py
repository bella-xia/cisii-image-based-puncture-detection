import time
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from ultralytics import YOLO
from image_conversion_without_using_ros import image_to_numpy
from image_conversion_without_using_ros import numpy_to_image

global iOCT_frame

PUNCTURE_THRESHOLD = 0.8

def convert_ros_to_numpy(image_message):
    global iOCT_frame
    if iOCT_frame is None:
        iOCT_frame = image_to_numpy(image_message)
        iOCT_frame = cv2.resize(iOCT_frame, (640, 480))

if __name__ == "__main__": 
    rospy.init_node('image_puncture_detection', anonymous=True)

    ros_img_msg = rospy.wait_for_message("/decklink/camera/image_raw", Image)
    iOCT_frame = image_to_numpy(ros_img_msg)
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))
    print("DIRECTLY FROM ROS")
    print(iOCT_frame.dtype)
    print(iOCT_frame.shape)
    print(iOCT_frame.max())
    
    print("SAVE AND READ")
    cv2.imwrite("test.png", iOCT_frame)
    read_image = cv2.imread("test.png")
    print(read_image.dtype)
    print(read_image.shape)
    print(read_image.max())

    print((iOCT_frame == read_image).all())
