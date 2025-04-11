import numpy as np


class KalmanFilter:

    def __init__(self, alpha=0.01):
        # State transition matrix
        self.F_base = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        # Observation matrix
        self.H_base = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        # Initial state (x, x_velocity, y, y_velocity)
        self.x = np.zeros(4)
        # Initial process covariance
        self.P = np.eye(4) * 100  # Large uncertainty
        # Measurement noise covariance (x and y)
        self.V = np.eye(2)
        # Initial adaptive process noise
        self.Q = np.eye(4)
        # Identity matrix
        self.G_base = np.eye(4)
        # Adaptive process noise update factor
        self.alpha = alpha

    def filter_instance(self, z):
        """
        :param z: Measurement at point t (observed [x, y])
        :return: Updated state (x_t|t)
        """
        # Prediction step
        x_pred = self.F_base @ self.x
        P_pred = self.F_base @ self.P @ self.F_base.T + self.Q

        # Compute residual (innovation)
        epsilon = z - (self.H_base @ x_pred)

        # Compute Kalman gain
        S = self.H_base @ P_pred @ self.H_base.T + self.V
        K = P_pred @ self.H_base.T @ np.linalg.inv(S)

        # Update step
        self.x = x_pred + K @ epsilon
        self.P = (np.eye(self.P.shape[0]) - K @ self.H_base) @ P_pred

        # Adaptive process noise update
        innovation_x, innovation_y = epsilon[0], epsilon[1]
        self.Q = (1 - self.alpha) * self.Q + self.alpha * np.diag(
            [innovation_x**2, self.x[1] ** 2, innovation_y**2, self.x[3] ** 2]
        )

        return self.x
