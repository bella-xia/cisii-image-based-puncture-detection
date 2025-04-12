import numpy as np
import torch, re
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from _utils_model.image_based_util_kalman import KalmanFilter
from _utils_model.image_based_util_dbscan import filter_small_segments_with_dbscan


class ImageProcessor:

    def __init__(
        self,
        model_path,
        img_w=640,
        img_h=480,
        outlier=10000,
        y_border=480,
        velocity_threshold=20,
        acceleration_threshod = 10,
        stop_robot_after_detection=1,  # unit of seconds
        rate_threshold=None,
        mode="velocity",
    ):

        assert (
            mode == "normal" or mode == "kalman" or mode == "velocity"
        ), "mode should be normal, kalman or velocity"

        # load model
        self.model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1)
        # TODO: try to load the model with GPU
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        # self.device = torch.device("cuda" if torch.cuda.is_available() eldse "cpu")
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # get transform
        search_augment = re.search(r"augmented", model_path)
        if search_augment:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = None

        # initialize kalman filter
        self.kalman = KalmanFilter()

        # initialize data structure for processed image data
        self.puncture_detect = False
        self.protection_mode = False
        self.mode = mode

        self.ds_arr = []

        self.px, self.py = img_w, img_h
        self.kpx, self.kpy = img_w, img_h
        self.vel = 0
        self.img_w, self.img_h = img_w, img_h

        self.first_n, self.detected = 0, 0
        self.stop_robot_after_detection = stop_robot_after_detection
        self.outlier, self.y_border = outlier, y_border
        self.velocity_threshold, self.acceleration_threshold = velocity_threshold, acceleration_threshod
        self.rate_threshold = rate_threshold

    def generate_auxiliary_data(self, image):

        if not self.transform:
            tensor_img = (
                torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
        else:
            tensor_img = self.transform(image=image)["image"].unsqueeze(0)

        tensor_img = tensor_img.to(self.device)
        if tensor_img.shape[-1] != self.img_w or tensor_img.shape[-2] != self.img_h:
            mask = np.zeros((self.img_h, self.img_w))
            print(
                f"invalid image size {tensor_img.shape[-1]}x{tensor_img.shape[-2] if tensor_img != None else '<null>'}."
            )
            return mask, -1, -1

        with torch.no_grad():
            results = self.model.predict(tensor_img)
        mask = results.sigmoid().detach().cpu().numpy()[0, 0, :, :]
        mask = filter_small_segments_with_dbscan(mask)

        y_indices, x_indices = np.where(mask > 0.1)
        if len(y_indices) == 0:
            pos_x, pos_y = -1, -1
        else:
            pos_y = np.min(y_indices)
            pos_x = np.min(x_indices[y_indices == pos_y])

        return mask, pos_x, pos_y

    def serialized_processing(self, new_image):

        mask_msg, px_t, py_t = self.generate_auxiliary_data(new_image)

        if py_t == -1 or px_t == -1 or py_t > self.y_border:
            self.protection_mode = False
            self.first_n = 0

            return [px_t, py_t, -1, -1, -1, -1], mask_msg, 0, -1

        x_update = self.kalman.filter_instance(np.array([px_t, py_t]))
        kpx_t, kpy_t = x_update[0], x_update[2]
        puncture_flag = False

        if self.px == self.img_w or self.py == self.img_h:
            # first instance where the needle appears
            # go to protection mode
            self.protection_mode = True
            ds_t = 0
        else:
            dx_t, dy_t = px_t - self.px, py_t - self.py
            sign = 1 if (dx_t < 0 and dy_t < 0) else -1
            ds_t = sign * np.sqrt(dx_t**2 + dy_t**2)

        if self.protection_mode:
            self.first_n += 1
            if self.first_n > 30:
                self.protection_mode = False
                self.first_n = 0

        kdx_t, kdy_t = kpx_t - self.kpx, kpy_t - self.kpy
        sign = 1 if (kdx_t < 0 and kdy_t < 0) else -1
        kds_t = sign * np.sqrt(kdx_t**2 + kdy_t**2)
        kvx_t, kvy_t = x_update[1], x_update[3]
        sign = 1 if (kvx_t < 0 and kvy_t < 0) else -1
        kv_t = sign * np.sqrt(kvx_t**2 + kvy_t**2)
        ka_t = abs(kv_t) - self.vel

        if self.rate_threshold:
            if self.mode == "normal":
                self.ds_arr.append(ds_t)
            elif self.mode == "kalman":
                self.ds_arr.append(kds_t)
            elif self.mode == "velocity":
                self.ds_arr.append(kv_t)

        if self.puncture_detect:
            self.detected += 1
            if self.detected > 30 * self.stop_robot_after_detection:
                self.puncture_detect = False
                puncture_flag = False
                self.protection_mode = True
                self.first_n = 0
            else:
                puncture_flag = True
        elif (
            # not self.protection_mode
            # and 
            max([ds_t, kds_t, kv_t]) < self.outlier
            and (not self.rate_threshold or len(self.ds_arr) > 2)
        ):

            identifier = 0
            if self.mode == "normal":
                identifier = ds_t
            elif self.mode == "kalman":
                identifier = kds_t
            else:
                identifier = kv_t

            if (
                # not self.puncture_detect and
                identifier
                > self.velocity_threshold
                and ka_t > self.acceleration_threshold
                # and (
                #     not self.rate_threshold
                #     or identifier
                #     > statistics.mean(self.ds_arr[:-1]) * self.rate_threshold
                # )
            ):
                self.puncture_detect = True
                print("puncture detected")
                puncture_flag = True
                self.detected = 0

        self.px, self.py = px_t, py_t
        self.kpx, self.kpy = kpx_t, kpy_t
        self.vel = abs(kv_t)
        return (
            [
                px_t,
                py_t,
                float(x_update[0]),
                float(x_update[2]),
                float(x_update[1]),
                float(x_update[3]),
            ],
            mask_msg,
            1 if puncture_flag else 0,
            ka_t
        )
