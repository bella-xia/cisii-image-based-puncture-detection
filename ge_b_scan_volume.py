import rospy, cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from leica_engine import LeicaEngine

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

    # b_scan_publisher = rospy.Publisher("/oct_b_scan", Image, queue_size=1)
    # cv_bridge = CvBridge()
    # rospy.init_node("b_scan_publisher", anonymous=True)
    print("B scan publisher initialized")

    if not rospy.is_shutdown():
            # print("in the loop")
            b_scan_img_volume = leica_reader.fast_get_b_scan_volume()
            if b_scan_img_volume is not None:
                print(b_scan_img_volume.shape)

            cv2.imshow(b_scan_img_volume[0])