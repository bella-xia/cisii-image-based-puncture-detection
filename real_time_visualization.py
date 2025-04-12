import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import numpy as np
import re, cv2, time

from std_msgs.msg import Int32, Float64
from sensor_msgs.msg import Image

from _utils_rospy.subscriber_module import SubRosTopic, RosTopicSubscriber
from _utils_image.image_conversion_without_using_ros import image_to_numpy


class ROSVisualizationModule:
    def __init__(self, color_priorities=["w", "r", "g", "y", "b"]):
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Real-Time Visualization")
        self.win.resize(1200, 600)
        self.win.show()

        # Image views
        self.image_view = pg.ImageItem()
        self.mask_view = pg.ImageItem()
        self.bscan_view = pg.ImageItem()

        view1 = self.win.addViewBox(row=0, col=0)
        view1.addItem(self.image_view)

        view2 = self.win.addViewBox(row=0, col=1)
        view2.addItem(self.mask_view)

        view3 = self.win.addViewBox(row=0, col=2)
        view3.addItem(self.bscan_view)

        self.scatter_px = pg.ScatterPlotItem(
            pen=pg.mkPen(None), brush=pg.mkBrush("r"), size=10
        )
        self.scatter_mask = pg.ScatterPlotItem(
            pen=pg.mkPen(None), brush=pg.mkBrush("r"), size=10
        )
        view1.addItem(self.scatter_px)
        view2.addItem(self.scatter_mask)

        title_topic_map = {
            "image_based_tip_velocity": "/image_model/Velocity",
            "robot_velocity": "/eye_robot/robotVelZ_Global",
            "needle_tip_force_dot": "/eye_robot/TipForceNormGlobal",
            "tip_x_pos": "/image_model/SegmentPosX",
            "tip_y_pos": "/image_model/SegmentPosY",
            "x_label": "/eye_robot/TimePuncture_Global",
            "raw_image": "/decklink/camera/image_raw",
            "mask_image": "/image_model/MaskImage",
            "b_scan": "/b_scan/top_frame",
        }
        self.data_meta = [
            "image_based_tip_velocity",
            "robot_velocity",
            "needle_tip_force_dot",
        ]
        self.image_meta = [
            ("raw_image", self.image_view),
            ("mask_image", self.mask_view),
            ("b_scan", self.bscan_view),
        ]

        subscribers = []

        self.plots = []
        self.lines = [[] for _ in range(len(title_topic_map) - 1)]  # x/y/mag per plot

        for col, (title, topic) in enumerate(title_topic_map.items()):
            if col < 5:  # plottable data
                subscribers.append(SubRosTopic(topic, Float64, title + "_sub", title))
                if col >= 3:  # this is for tip information
                    continue

                # the others are plotting data
                p = self.win.addPlot(row=1, col=col, title=re.sub(r"_", r" ", title))
                # p.setYRange(-5, 30)
                p.addLegend()
                self.plots.append(p)
                if col == 2:  # force tip
                    self.lines[col].append(
                        p.plot(pen=color_priorities[0], name="force dot")
                    )
                else:
                    self.lines[col].append(
                        p.plot(pen=color_priorities[0], name="velocity")
                    )
                    self.lines[col].append(
                        p.plot(pen=color_priorities[1], name="acceleration")
                    )

            elif col > 5:  # image data
                subscribers.append(SubRosTopic(topic, Image, title + "_sub", title))

        self.subscriber_manager = RosTopicSubscriber(subscribers)

        self.x_data = []
        self.x_label = []
        self.y_data = [[] for _ in range(5)]

        # take note of the last
        self.last_data = [[0], [0]]
        self.num_frames = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

    def update_data(self):

        # update all images
        for attr_name, view in self.image_meta:
            img_msg = self.subscriber_manager.get_data(attr_name)

            if img_msg is not None:
                img = image_to_numpy(img_msg)

                # assuming at this point we do not need this....
                # if image.dtype == np.float32 or image.dtype == np.float64:
                #     image = (
                #         255 * (image - np.min(image)) / (np.max(image) - np.min(image))
                #     ).astype(np.uint8)
                # elif image.dtype != np.uint8:
                #     image = image.astype(np.uint8)
                if attr_name == "raw_image" and img is not None:
                    img = np.transpose(np.flipud(img), (1, 0, 2))
                else:
                    img = np.transpose(np.flipud(img), (1, 0))

                img = cv2.resize(img, (640, 480))
                view.setImage(img, autoLevels=False)

        px = self.subscriber_manager.get_data("tip_x_pos")
        py = self.subscriber_manager.get_data("tip_y_pos")
        if px and py:
            px, py = int(px.data), int(py.data)
            self.scatter_px.setData([px], [480 - py])
            self.scatter_mask.setData([px], [480 - py])

        self.x_data.append(self.num_frames)
        self.x_label.append(self.subscriber_manager.get_data("x_label"))
        self.num_frames += 1

        for col, title in enumerate(self.data_meta):

            new_instance = self.subscriber_manager.get_data(title)
            new_data = new_instance.data

            if new_data:
                if col == 2:
                    self.y_data[-1].append(new_data)
                else:
                    self.y_data[col * 2].append(new_data)
                    self.y_data[col * 2 + 1].append(
                        abs(new_data) - self.last_data[col][-1]
                    )

    def update(self):
        if not self.x_data:
            return

        data_len = min(50, len(self.x_data))

        for i in range(3):
            self.lines[i][0].setData(
                self.x_data[-data_len:], self.y_data[i * 2][-data_len:]
            )
            if i != 2:
                self.lines[i][1].setData(
                    self.x_data[-data_len:], self.y_data[i * 2 + 1][-data_len:]
                )


# Example use:
# visualizer = VisualizationModulePG()
# while True:
#     img, mask, data = ...  # your data
#     visualizer.add_data(img, mask, data)
#     QtGui.QApplication.processEvents()

if __name__ == "__main__":
    visualizer = ROSVisualizationModule()
import time


def update():
    # Define the logic for your module update
    print("Updating...")
    fps = 30
    interval = 1 / fps

    try:
        while True:
            start_time = time.time()
            visualizer.update_data()
            QtGui.QApplication.processEvents()
            elapsed_time = time.time() - start_time
            time_to_sleep = max(0, interval - elapsed_time)
            time.sleep(time_to_sleep)

    except KeyboardInterrupt:
        print("Encountered keyboard interrupt. Quiting")

    finally:
        visualizer.app.quit()
        print(f"successfully quited visualization")
