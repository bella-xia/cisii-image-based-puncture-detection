import os
import cv2
from ultralytics import YOLO
import time
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from image_conversion_without_using_ros import image_to_numpy
import numpy as np
import torch
import csv

PUNCTURE_THRESHOLD = 0.8

needle_det_model = YOLO("crop_weights7.pt")

puncture_det_model = YOLO("classification_weights.pt")

def find_and_crop(image):
    start = time.perf_counter()
    result = needle_det_model(image)
    print(f"end: {time.perf_counter() - start:.4f}")
    try:
        x1, y1, x2, y2 = result[0].boxes.xyxy.tolist()[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1, y1, x2, y2)
    except IndexError:
        print("No needle detected")
        return image
    cropped_image = cv2.imread(image)[y1:y2, x1:x2]
    return cropped_image

def detect_puncture(cropped_image):
    # Predict with the model
    results = puncture_det_model(cropped_image)  # predict on an image
    puncture_prob = results[0].probs.data.cpu().numpy()[2]
    print("PUNCTURE PROBABILITY:", puncture_prob)
    if puncture_prob >= PUNCTURE_THRESHOLD:
        puncture_flag = True
    else:
        puncture_flag = False
    return puncture_flag


import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# Define the test folder path
test_folders = "_recordings/Egg06_04_TP_good_quality"

# # Iterate through folders in the test folder
# for folder in os.listdir(test_folders):
#     folder_path = os.path.join(test_folders, folder)
    
#     if os.path.isdir(folder_path):  # Ensure it's a directory
#         # Create a CSV file for each folder
#         csv_filename = f"{folder}_puncture_detection_results.csv"
#         csv_filepath = os.path.join(test_folders, csv_filename)
        
#         with open(csv_filepath, mode='w', newline='') as csv_file:
#             # Define the CSV writer
#             fieldnames = ['image_path', 'puncture_flag']
#             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
#             # Write the header
#             writer.writeheader()

#             # Iterate through images in the folder
#             for image in os.listdir(folder_path):
#                 if image.startswith("iOCT"):
#                     # Full image path
#                     image_path = os.path.join(folder_path, image)

#                     # Process the image: Crop and detect puncture
#                     cropped_image = find_and_crop(image_path)
#                     puncture_flag = detect_puncture(cropped_image)

#                     # Write the result to the folder-specific CSV file
#                     writer.writerow({'image_path': image_path, 'puncture_flag': puncture_flag})
        

folder = "Egg06_04_TP_good_quality"
    
if os.path.isdir(test_folders):  # Ensure it's a directory
    # Create a CSV file for each folder
    csv_filename = f"test_puncture_detection_results.csv"
    csv_filepath = os.path.join(csv_filename)
    
    with open(csv_filepath, mode='w', newline='') as csv_file:
        # Define the CSV writer
        fieldnames = ['image_path', 'puncture_flag']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()

        # Iterate through images in the folder
        for image in os.listdir(test_folders):
            if image.startswith("iOCT"):
                # Full image path
                image_path = os.path.join(test_folders, image)

                # Process the image: Crop and detect puncture
                cropped_image = find_and_crop(image_path)
                puncture_flag = detect_puncture(cropped_image)

                # Write the result to the folder-specific CSV file
                writer.writerow({'image_path': image_path, 'puncture_flag': puncture_flag})
                break
            

# for image_file_name in os.listdir("path/to/images/folder"):
#     image = cv2.imread("path/to/images/folder/" + image_file_name)
#     cropped_image = find_and_crop(image)
#     puncture_flag = detect_puncture(cropped_image)
#     print(puncture_flag)


# image_path = "/home/peiyao/Desktop/Mojtaba/vein-puncture-detection-egg/_recordings/Egg13_08_TP_bleeding/iOCT_image_448.3174460000.jpg"

# image = cv2.imread(image_path)
# cropped_image = find_and_crop(image)
# puncture_flag = detect_puncture(cropped_image)
# print(puncture_flag)