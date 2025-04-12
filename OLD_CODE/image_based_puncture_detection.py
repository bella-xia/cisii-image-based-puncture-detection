# python lib imports
import time, cv2
from ultralytics import YOLO

# rospy imports
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

# self-defined function imports
from _utils_image.image_conversion_without_using_ros import image_to_numpy
from _utils_image.image_conversion_without_using_ros import numpy_to_image

global iOCT_frame

PUNCTURE_THRESHOLD = 0.8


def convert_ros_to_numpy(image_message):
    global iOCT_frame

    iOCT_frame = image_to_numpy(image_message)
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))


def find_and_crop(microscope_image):
    result = needle_det_model(microscope_image)
    try:
        x1, y1, x2, y2 = result[0].boxes.xyxy.tolist()[0]
    except IndexError:
        print("No needle detected")
        image_message = numpy_to_image(microscope_image, encoding="rgb8")
        pub_cropped_image.publish(image_message)
        return microscope_image
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped_image = microscope_image[y1:y2, x1:x2]
    cropped_image_message = numpy_to_image(cropped_image, encoding="rgb8")
    pub_cropped_image.publish(cropped_image_message)
    return cropped_image


def detect_puncture(cropped_image):
    # Predict with the model
    results = puncture_det_model(cropped_image)  # predict on an image
    puncture_prob = results[0].probs.data.cpu().numpy()[2]
    pub_puncture_prob.publish(puncture_prob)
    print("PUNCTURE PROBABILITY:", puncture_prob)
    if puncture_prob >= PUNCTURE_THRESHOLD:
        puncture_flag = True
    else:
        puncture_flag = False
    return puncture_flag


if __name__ == "__main__":
    # # cropping model
    # needle_det_model = YOLO(
    #     "crop_weights7.pt"  # TODO: change path to trained bounding box model
    # )
    # # detection classification model
    # puncture_det_model = YOLO(
    #     "classification_weights.pt" # TODO: change path to trained classification model
    # )

    pub_puncture_flag = rospy.Publisher("PunctureFlagImage", Bool, queue_size=1)
    pub_puncture_prob = rospy.Publisher(
        "PunctureProbabilityImage", Float32, queue_size=1
    )
    pub_cropped_image = rospy.Publisher("CroppedImage", Image, queue_size=1)

    iOCT_camera_sub = rospy.Subscriber(
        "/decklink/camera/image_raw", Image, convert_ros_to_numpy
    )

    rospy.init_node("image_puncture_detection", anonymous=True)
    time.sleep(0.5)
    while not rospy.is_shutdown():
        # cropped_image = find_and_crop(iOCT_frame)
        # puncture_flag = detect_puncture(cropped_image)
        pub_puncture_flag.publish(False)
