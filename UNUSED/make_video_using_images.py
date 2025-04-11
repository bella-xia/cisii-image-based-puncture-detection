#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import re

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


image_folder = '/media/peiyao/SSD1T/mojtaba/recordings/Mojtaba_test3_Oct9/Egg14_05_TP_HighQuality_video_Demir'
video_name = 'concatenate_Demir.mkv'

images = sorted([img for img in os.listdir(image_folder) if img.endswith('.jpg')], key=natural_sort_key)

# images = []
# for i in range(100, 2000):
#     s = 'iOCT_image_{:06d}'.format(i)+'.jpg'
#     images.append(s)
# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
# print("frame = ", frame)
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 16, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
video.release()



############################################################################
#Another method based on image numbes 

# import cv2
# import os


# image_folder = '35_TP_double_puncture_bleeding'
# video_name = 'concatenate.mkv'
# images = []
# for i in range(295, 795):
#     s = 'iOCT_image_{:06d}'.format(i)+'.jpg'
#     images.append(s)
# # images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# print(images)
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# # print("frame = ", frame)
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 16, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# # cv2.destroyAllWindows()
# video.release()

