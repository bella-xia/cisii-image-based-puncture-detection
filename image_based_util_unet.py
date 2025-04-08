import numpy as np
import torch, statistics, re
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from image_based_util_dbscan import filter_small_segments_with_dbscan

# from image_conversion_without_using_ros import numpy_to_image
from image_based_util_kalman import KalmanFilter


class ImageProcessor:

    def __init__(
        self,
        model_path,
        img_w=640,
        img_h=480,
        outlier=60,
        y_border=430,
        numeric_threshold=10,
        rate_threshod=None,
        mode="velocity",
        seg_posx_publisher=None,
        seg_posy_publisher=None,
        kal_posx_publisher=None,
        kal_posy_publisher=None,
        kal_velx_publisher=None,
        kal_vely_publisher=None,
        mask_publisher=None,
        puncture_flag_publisher=None,
        logger=None,
    ):

        assert (
            mode == "normal" or mode == "kalman" or mode == "velocity"
        ), "mode should be normal, kalman or velocity"

        # load model
        self.model = smp.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.img_w, self.img_h = img_w, img_h

        self.first_n = 0
        self.outlier, self.y_border = outlier, y_border
        self.numeric_threshold = (numeric_threshold,)
        self.rate_threshold = rate_threshod

        # define all publishers
        self.seg_posx_publisher, self.seg_posy_publisher = (
            seg_posx_publisher,
            seg_posy_publisher,
        )
        self.kal_posx_publisher, self.kal_posy_publisher = (
            kal_posx_publisher,
            kal_posy_publisher,
        )
        self.kal_velx_publisher, self.kal_vely_publisher = (
            kal_velx_publisher,
            kal_vely_publisher,
        )
        self.mask_publisher, self.puncture_flag_publisher = (
            mask_publisher,
            puncture_flag_publisher,
        )
        self.logger = logger

    def try_publish(self, publishers, data, spec_strs=None):
        if len(publishers) != len(data):
            if self.logger:
                self.logger.info(
                    "Mismatch between number of publishers and data instances. Skipping publishing."
                )
            return

        if not spec_strs:
            spec_strs = [""] * len(publishers)

        for publisher, data_instance, spec_str in zip(publishers, data, spec_strs):
            if publisher:
                try:
                    publisher.publish(data_instance)
                except Exception as e:
                    if self.logger:
                        self.logger.info(
                            f"Failed to publish data {data_instance} at publisher-{spec_str}: {str(e)}"
                        )
                    else:
                        pass
            else:
                if self.logger:
                    self.logger.info(
                        f"Publisher-{spec_str} is not defined, skipping publishing. Expected to publish {data_instance}"
                    )
                else:
                    continue

    def generate_auxiliary_data(self, image):
        if not self.transform:
            tensor_img = (
                torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
        else:
            tensor_img = self.transform(image=image)["image"].unsqueeze(0)
        tensor_img = tensor_img.to(self.device)

        with torch.no_grad():
            results = self.model.predict(tensor_img)
        mask = results.sigmoid().detach().cpu().numpy()[0, 0, :, :]
        mask = filter_small_segments_with_dbscan(mask)
        # mask_msg = numpy_to_image(mask.astype(np.uint8), encoding="mono8")
        # self.try_publish([self.mask_publisher], [mask_msg])
        y_indices, x_indices = np.where(mask > 0.1)
        if len(y_indices) == 0:
            top, left = -1, -1
        else:
            top = np.min(y_indices)
            left = np.min(x_indices[y_indices == top])

        return left, top

    def serialized_processing(self, new_image):

        px_t, py_t = self.generate_auxiliary_data(new_image)

        self.try_publish(
            [self.seg_posx_publisher, self.seg_posy_publisher],
            [px_t, py_t],
            spec_strs=["seg_posx", "seg_posy"],
        )

        if py_t == -1 or px_t == -1 or py_t > self.y_border:
            self.protection_mode = False
            self.first_n = 0

            self.try_publish(
                [
                    self.kal_posx_publisher,
                    self.kal_posy_publisher,
                    self.kal_velx_publisher,
                    self.kal_vely_publisher,
                    self.puncture_flag_publisher,
                ],
                [-1, -1, -1, -1, False],
                spec_strs=[
                    "kal_posx",
                    "kal_posy",
                    "kal_velx",
                    "kal_vely",
                    "puncture_flag",
                ],
            )

            return

        x_update = self.kalman.filter_instance(np.array([px_t, py_t]))
        kpx_t, kpy_t = x_update[0], x_update[2]

        self.try_publish(
            [
                self.kal_posx_publisher,
                self.kal_posy_publisher,
                self.kal_velx_publisher,
                self.kal_vely_publisher,
            ],
            [
                float(x_update[0]),
                float(x_update[2]),
                float(x_update[1]),
                float(x_update[3]),
            ],
            spec_strs=["kal_posx", "kal_posy", "kal_velx", "kal_vely"],
        )

        if self.px == self.img_w or self.py == self.img_h:
            # first instance where the needle appears
            # go to protection mode
            self.protection_mode = True
        else:
            dx_t, dy_t = px_t - self.px, py_t - self.py
            sign = 1 if (dx_t < 0 and dy_t < 0) else -1
            ds_t = sign * np.sqrt(dx_t**2 + dy_t**2)

        if self.protection_mode:
            self.first_n += 1
            if self.first_n > 90:
                self.protection_mode = False
                self.first_n = -1

        kdx_t, kdy_t = kpx_t - self.kpx, kpy_t - self.kpy
        sign = 1 if (kdx_t < 0 and kdy_t < 0) else -1
        kds_t = sign * np.sqrt(kdx_t**2 + kdy_t**2)
        kvx_t, kvy_t = x_update[1], x_update[3]
        sign = 1 if (kvx_t < 0 and kvy_t < 0) else -1
        kv_t = sign * np.sqrt(kvx_t**2 + kvy_t**2)

        if self.rate_threshold:
            if self.mode == "normal":
                self.ds_arr.append(ds_t)
            elif self.mode == "kalman":
                self.ds_arr.append(kds_t)
            elif self.mode == "velocity":
                self.ds_arr.append(kv_t)

        if (
            not self.protection_mode
            and max([ds_t, kds_t, kv_t]) < self.outlier
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
                not self.puncture_detect
                and identifier > self.numeric_threshold
                and (
                    not self.rate_threshold
                    or identifier
                    > statistics.mean(self.ds_arr[:-1]) * self.rate_threshold
                )
            ):
                self.puncture_detect = True
                self.try_publish(
                    [self.puncture_flag_publisher], [True], spec_strs=["puncture_flag"]
                )
            else:
                self.try_publish(
                    [self.puncture_flag_publisher], [False], spec_strs=["puncture_flag"]
                )
        else:
            self.try_publish(
                [self.puncture_flag_publisher], [False], spec_strs=["puncture_flag"]
            )

        self.px, self.py = px_t, py_t
        self.kpx, self.kpy = kpx_t, kpy_t
