import matplotlib.pyplot as plt
import numpy as np


class VisualizationModule:

    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))

        self.axes[0][0].set_title("Image")
        self.axes[0][0].axis("off")
        self.axes[0][1].set_title("Mask Image")
        self.axes[0][1].axis("off")
        self.axes[0][2].axis("off")

        (self.line1_1,) = self.axes[1][0].plot([], [], label="x", color="blue")
        (self.line1_2,) = self.axes[1][0].plot([], [], label="y", color="red")
        (self.line1_3,) = self.axes[1][0].plot([], [], label="mag", color="green")
        self.axes[1][0].set_title("Tip Position")
        self.axes[1][0].set_ylim(-5, 30)
        self.axes[1][0].set_xlim(0, 2000)
        self.axes[1][0].set_autoscaley_on(False)
        self.axes[1][0].legend()

        (self.line2_1,) = self.axes[1][1].plot([], [], label="x", color="blue")
        (self.line2_2,) = self.axes[1][1].plot([], [], label="y", color="red")
        (self.line2_3,) = self.axes[1][1].plot([], [], label="mag", color="green")
        self.axes[1][1].set_title("Post-Kalman Tip Position")
        self.axes[1][1].set_ylim(-5, 30)
        self.axes[1][1].set_xlim(0, 2000)
        self.axes[1][1].set_autoscaley_on(False)
        self.axes[1][1].legend()

        (self.line3_1,) = self.axes[1][2].plot([], [], label="x", color="blue")
        (self.line3_2,) = self.axes[1][2].plot([], [], label="y", color="red")
        (self.line3_3,) = self.axes[1][2].plot([], [], label="mag", color="green")
        self.axes[1][2].set_title("Post-Kalman Tip Velocity")
        self.axes[1][2].set_ylim(-5, 30)
        self.axes[1][2].set_xlim(0, 2000)
        self.axes[1][2].set_autoscaley_on(False)
        self.axes[1][2].legend()

        self.x_data = []
        self.y_data = [[] for _ in range(9)]
        self.pos_data = [[] for _ in range(4)]
        self.pos_data[0].append(640)
        self.pos_data[2].append(640)
        self.pos_data[1].append(480)
        self.pos_data[3].append(480)
        self.num_frames = 0

        self.scatter_px = self.axes[0][0].scatter([], [], color="blue", s=10)
        self.scatter_mask_px = self.axes[0][1].scatter([], [], color="blue", s=10)

    def add_data(self, image, mask, data):

        px, py, kpx, kpy, vx, vy = data[0], data[1], data[2], data[3], data[4], data[5]

        self.axes[0][0].imshow(image)
        self.axes[0][1].imshow(mask, cmap="gray")
        self.scatter_px.set_offsets([[px, py]])
        self.scatter_mask_px.set_offsets([[px, py]])

        dpx, dpy = px - self.pos_data[0][-1], py - self.pos_data[1][-1]
        dkpx, dkpy = kpx - self.pos_data[2][-1], kpy - self.pos_data[3][-1]
        dp_avg, dkp_avg, v_avg = (
            np.sqrt(dpx**2 + dpy**2),
            np.sqrt(dkpx**2 + dkpy**2),
            np.sqrt(vx**2 + vy**2),
        )

        self.x_data.append(self.num_frames)
        self.num_frames += 1

        self.y_data[0].append(dpx)
        self.y_data[1].append(dpy)
        self.y_data[2].append(dp_avg)
        self.y_data[3].append(kpx)
        self.y_data[4].append(kpy)
        self.y_data[5].append(dkp_avg)
        self.y_data[6].append(vx)
        self.y_data[7].append(vy)
        self.y_data[8].append(v_avg)

        self.line1_1.set_data(self.x_data, self.y_data[0])
        self.line1_2.set_data(self.x_data, self.y_data[1])
        self.line1_3.set_data(self.x_data, self.y_data[2])
        self.line2_1.set_data(self.x_data, self.y_data[3])
        self.line2_2.set_data(self.x_data, self.y_data[4])
        self.line2_3.set_data(self.x_data, self.y_data[5])
        self.line3_1.set_data(self.x_data, self.y_data[6])
        self.line3_2.set_data(self.x_data, self.y_data[7])
        self.line3_3.set_data(self.x_data, self.y_data[8])

        for idx in range(3):
            self.axes[1][idx].relim()

        for idx in range(4):
            self.pos_data[idx].append(data[idx])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
