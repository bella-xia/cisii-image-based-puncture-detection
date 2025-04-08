from ultralytics import YOLO
import os
import cv2

model = YOLO('/home/demir/Desktop/vein-puncture-detection-egg/runs/detect/train7/weights/best.pt')  # TODO: change path to trained bounding box model

images_path = "data/Labeled_Images"

new_dataset_path = "data/cropped_images"

for label in os.listdir(images_path):
    for image in os.listdir(os.path.join(images_path, label)):
        result = model(os.path.join(images_path, label, image))

        try:
            x1, y1, x2, y2 = result[0].boxes.xyxy.tolist()[0]
        except IndexError:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_image = cv2.imread(os.path.join(images_path, label, image))[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(new_dataset_path, "train", label, image), cropped_image)
