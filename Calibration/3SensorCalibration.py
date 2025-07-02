import numpy as np
import cv2
import os
from render_utils import mapping_data_collection, get_list


sensor_data_root = './sensor1/'
BallRad = 3.5
pixel_per_mm = 0.1

ref_path = os.path.join(sensor_data_root, 'ref.png')
ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

sample_path = os.path.join(sensor_data_root, 'cali_ball/ball.png')
sample_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

gray_list, depth_list = mapping_data_collection(sample_img, ref_img, BallRad, pixel_per_mm)
gray_list = np.array(gray_list)
depth_list = np.array(depth_list)
Pixel_to_Depth = get_list(gray_list, depth_list)
np.save(sensor_data_root + 'Pixel_to_Depth.npy', Pixel_to_Depth)
