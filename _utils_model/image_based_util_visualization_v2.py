import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtWidgets import QGraphicsRectItem
from PyQt5.QtGui import QBrush, QColor
import numpy as np


class VisualizationModulePG:
    def __init__(self):
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
            pen=pg.mkPen(None), brush=pg.mkBrush("g"), size=10
        )
        self.scatter_mask = pg.ScatterPlotItem(
            pen=pg.mkPen(None), brush=pg.mkBrush("g"), size=10
        )

        view1.addItem(self.scatter_px)
        view2.addItem(self.scatter_mask)
        self.label = pg.TextItem(text="No Puncture", color="green", anchor=(0.5, 0.5))
        view2.addItem(self.label)
        self.label.setPos(75, 100)

        self.bounding_box = QGraphicsRectItem(50, 50, 100, 20)
        self.bounding_box.setPen(pg.mkPen(color="green", width=2))
        green_qcolor = QColor("green")
        green_qcolor.setAlpha(255)
        self.bounding_box.setBrush(QBrush(green_qcolor))
        view2.addItem(self.bounding_box)

        # Real-time plots
        self.plots = []
        self.lines = [[] for _ in range(3)]  # x/y/mag per plot

        titles = ["Tip Pos Δ", "Post-Kalman Pos", "Post-Kalman Velocity"]
        for col, title in enumerate(titles):
            p = self.win.addPlot(row=1, col=col, title=title)
            p.setYRange(-5, 30)
            p.addLegend()
            self.plots.append(p)

            # self.lines[col].append(p.plot(pen="y", name="x"))
            # self.lines[col].append(p.plot(pen="r", name="y"))
            self.lines[col].append(p.plot(pen="g", name="mag"))
        self.lines[-1].append(p.plot(pen="w", name="acc"))

        self.x_data = []
        self.y_data = [[] for _ in range(10)]
        self.pos_data = [[640], [480], [640], [480], [0]]
        self.num_frames = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

    def add_data(self, image, mask, data, bscan=None,
                flag=False):

        px, py, kpx, kpy, vx, vy, kv, ka = data

        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (
                255 * (image - np.min(image)) / (np.max(image) - np.min(image))
            ).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = np.transpose(np.flipud(image), (1, 0, 2))

        mask = (mask * 255).astype(np.uint8)
        mask = np.transpose(np.flipud(mask), (1, 0))

        self.image_view.setImage(image, autoLevels=False)  # OpenCV image is transposed
        self.mask_view.setImage(mask, autoLevels=False)

        if not flag:
            instance_color = "green"
            instance_text = "No Puncture"
        else:
            instance_color = "red"
            instance_text = "Puncture"
        qcolor = QColor(instance_color)
        qcolor.setAlpha(255)
        self.bounding_box.setPen(pg.mkPen(color=instance_color, width=2))
        self.bounding_box.setBrush(QBrush(qcolor))
        self.label.setColor(instance_color)
        self.label.setText(instance_text)

        if bscan is not None:
            bscan = bscan.astype(np.uint8)
            bscan = np.transpose(np.flipud(bscan), (1, 0))
            self.bscan_view.setImage(bscan)

        if px != -1 and py != -1:
            self.scatter_px.setData([px], [480 - py])
            self.scatter_mask.setData([px], [480 - py])
        else:
           self.scatter_px.setData([], [])
           self.scatter_mask.setData([], [])

        dpx = px - self.pos_data[0][-1]
        dpy = py - self.pos_data[1][-1]
        dkpx = kpx - self.pos_data[2][-1]
        dkpy = kpy - self.pos_data[3][-1]
        dp_avg = np.sqrt(dpx**2 + dpy**2)
        dkp_avg = np.sqrt(dkpx**2 + dkpy**2)
        v_avg = kv
        a_avg = ka

        self.x_data.append(self.num_frames)
        self.num_frames += 1

        self.y_data[0].append(dpx)
        self.y_data[1].append(dpy)
        self.y_data[2].append(dp_avg)
        self.y_data[3].append(dkpx)
        self.y_data[4].append(dkpy)
        self.y_data[5].append(dkp_avg)
        self.y_data[6].append(vx)
        self.y_data[7].append(vy)
        self.y_data[8].append(v_avg)
        self.y_data[9].append(a_avg)

        for i in range(4):
            self.pos_data[i].append(data[i])
        self.pos_data[-1].append(abs(v_avg))

    def update(self):
        if not self.x_data:
            return

        data_len = min(50, len(self.x_data))
        for i in range(3):  # for each plot
            # self.lines[i][0].setData(
            #     self.x_data[-data_len:], self.y_data[i * 3 + 0][-data_len:]
            # )
            # self.lines[i][1].setData(
            #     self.x_data[-data_len:], self.y_data[i * 3 + 1][-data_len:]
            # )
            self.lines[i][0].setData(
                self.x_data[-data_len:], self.y_data[i * 3 + 2][-data_len:]
            )
        self.lines[-1][1].setData(self.x_data[-data_len:], self.y_data[-1][-data_len:])


# Example use:
# visualizer = VisualizationModulePG()
# while True:
#     img, mask, data = ...  # your data
#     visualizer.add_data(img, mask, data)
#     QtGui.QApplication.processEvents()
