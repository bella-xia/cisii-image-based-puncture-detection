from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(data="data/puncture_detection.v4i.yolov8/data.yaml", device=[0,1])