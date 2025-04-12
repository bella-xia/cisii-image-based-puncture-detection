import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from leica_engine import LeicaEngine
from _utils_rospy.publisher_module import PubRosTopic, RosTopicPublisher

N_BSCANS = 5
DIMS = (0.1, 4)

if __name__ == "__main__":

    leica_reader = LeicaEngine(
        ip_address="192.168.1.75",
        n_bscans=N_BSCANS,
        xd=DIMS[0],
        yd=DIMS[1],
        zd=3.379,
    )

    bscan_mananger = RosTopicPublisher(
        [
            PubRosTopic("/b_scan/top_fram", Image, "topscan_pub"),
            PubRosTopic("/b_scan/bottom_frame", Image, "bottomscan_pub"),
        ]
    )

    cv_bridge = CvBridge()
    rospy.init_node("b_scan_publisher", anonymous=True)
    print("B scan publisher initialized")

    try:
        while not rospy.is_shutdown():
            b_scan_top_img = leica_reader.get_b_scan(frame_to_save=0)
            b_scan_bottom_img = leica_reader.get_b_scan(frame_to_save=1)
            if not b_scan_top_img is None:
                image_message = cv_bridge.cv2_to_imgmsg(b_scan_top_img * 255)
                bscan_mananger.publish_data(["topscan_pub"], [image_message])
            if not b_scan_bottom_img is None:
                image_message = cv_bridge.cv2_to_imgmsg(b_scan_bottom_img * 255)
                bscan_mananger.publish_data(["bottomscan_pub"], [image_message])

    except KeyboardInterrupt:
        print("Shutting down b scan publisher")
    finally:
        # Perform any necessary cleanup here
        rospy.signal_shutdown("Shutdown signal received")
