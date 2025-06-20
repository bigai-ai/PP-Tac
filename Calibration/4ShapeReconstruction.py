import cv2
# import Visualizer
import numpy as np
import json
import os
from render_utils import raw_image_2_height_map, height_map_2_depth_map, get_mask

sensor_data_root = './sensor1/'

Pixel_to_Depth = np.load(f"{sensor_data_root}Pixel_to_Depth.npy")
max_index = len(Pixel_to_Depth) - 1
lighting_threshold = 2
contact_gray_base = 20
depth_k = 100
MASK_CENTER = [329, 250]
MASK_RADIUS = 207

with open(os.path.join(sensor_data_root, 'calibration_new.cfg'), 'r') as f:
    cfg = json.load(f)
    print(cfg)
    f.close()

camera_matrix = np.array(cfg['camera_matrix'])
dist_coeffs = np.array(cfg['dist_coeffs'])
rvecs = [np.array(cfg['rvecs'])]
tvecs = [np.array(cfg['tvecs'])]
fx, fy = camera_matrix[0,0], camera_matrix[1,1]
cx, cy = camera_matrix[0,2], camera_matrix[1,2]
print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")


ref_path = os.path.join(sensor_data_root, 'ref.png')
ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

mask = get_mask(ref_img, MASK_CENTER, MASK_RADIUS)

depth_ref_path = os.path.join(sensor_data_root, 'ref_depth.png')
empty_depth_img = cv2.imread(depth_ref_path, cv2.IMREAD_UNCHANGED)

sample_path = os.path.join(sensor_data_root, 'cali_ball/test.png')
sample_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
sample_img = cv2.GaussianBlur(sample_img, (3, 3), 0)
sample_img[mask == 0] = 0

cv2.imshow('SampleImg', sample_img)

height_map = raw_image_2_height_map(sample_img, ref_img, Pixel_to_Depth, max_index, lighting_threshold)
depth_map = height_map_2_depth_map(height_map, contact_gray_base, depth_k)

depth = (empty_depth_img - depth_map)/1000
depth = cv2.GaussianBlur(depth, (5, 5), 5)
depth[mask == 0] = 0
cv2.imshow('DepthMap', depth_map)
cv2.imshow('Depth', depth)
cv2.waitKey(0)