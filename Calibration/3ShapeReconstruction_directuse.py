import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import yaml
from training.model import ConvMLP
import copy
import json
import time

def get_newest_ckpt():
    save_root = os.path.join('./training/checkpoint', '20241031')
    path= os.path.join(save_root, 'ckpt')

    newest_ckpt = None
    batch_num_newest = 0
    for ckpt in os.listdir(path):
        # print(ckpt)
        batch_num = int(ckpt.split('_')[1])
        if batch_num > batch_num_newest:
            batch_num_newest = batch_num
            newest_ckpt = ckpt
    return newest_ckpt

def get_mask(ref_img, MASK_CENTER, MASK_RADIUS):
    mask = np.zeros(ref_img.shape)
    print(f"mask shape: {mask.shape}")
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.linalg.norm([i-MASK_CENTER[0], j-MASK_CENTER[1]]) < MASK_RADIUS:
                mask[i, j] = 1
    return mask

def normalize_backward(img):
    img = img*20.0
    return img

if __name__ == '__main__':
    sensor_data_root = './sensor1/'
    with open(os.path.join(sensor_data_root, 'calibration_new.cfg'), 'r') as f:
        cfg = json.load(f)
        print(cfg)
        f.close()
    camera_matrix = np.array(cfg['camera_matrix'])
    dist_coeffs = np.array(cfg['dist_coeffs'])
    rvecs = [np.array(cfg['rvecs'])]
    tvecs = [np.array(cfg['tvecs'])]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    MASK_CENTER = [250, 329]
    MASK_RADIUS = 207

    device = torch.device("cuda:0")
    model = ConvMLP(1, ref=False)
    model.to(device)
    model.eval()
    newest_ckpt = get_newest_ckpt()
    rewake_ckpt = newest_ckpt
    REWAKE_CKPT_PATH = os.path.join('./training/checkpoint', '20241031', 'ckpt', rewake_ckpt)
    model.load_state_dict(torch.load(REWAKE_CKPT_PATH, map_location=device))

    empty_depth_img_path = os.path.join(sensor_data_root, 'ref_depth.png')
    empty_depth_img = cv2.imread(empty_depth_img_path, cv2.IMREAD_UNCHANGED)
    empty_depth_img = cv2.GaussianBlur(empty_depth_img, (5, 5), 5)

    cap = cv2.VideoCapture(1)

    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(3, 480)
    cap.set(4, 640)
    cap.set(5, 120)

    frame_count = 0
    ref_image = None
    ref_calculate_delta = None

    while True:
        initialize_time = time.time()
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32)
        frame = cv2.GaussianBlur(frame, (5, 5), 5)

        if not ret:
            print("can not get frame")
            break

        if frame_count == 0:
            ref_image = frame
            mask = get_mask(ref_image, MASK_CENTER, MASK_RADIUS)
            ref_image[mask == 0] = 0
            ref_calculate_delta = copy.deepcopy(ref_image)
            ref_image = ref_image / 255.0
            ref_image = torch.from_numpy(ref_image[:, :, np.newaxis]).permute(2, 0, 1)
            ref_image = ref_image.unsqueeze(0)
            ref_image = ref_image.to(device)
        else:
            img_cap = frame
            img_cap[mask == 0] = 0

            MIN = 3
            MAX = 47
            cap_gray_delta = ref_calculate_delta - img_cap
            cap_gray_delta = cap_gray_delta - MIN
            cap_gray_delta[cap_gray_delta < 0] = 0
            cap_gray_delta[cap_gray_delta > MAX] = MAX
            cap_gray_delta = cap_gray_delta / MAX
            cap_gray_delta_mask = np.zeros(cap_gray_delta.shape)
            cap_gray_delta_mask[cap_gray_delta != 0] = 1

            cap_gray_delta = torch.from_numpy(cap_gray_delta[:, :, np.newaxis]).permute(2, 0, 1)
            cap_gray_delta = cap_gray_delta.unsqueeze(0).to(device)
            pred_depth_delta = model(cap_gray_delta)

            torch.cuda.synchronize()
            pred_depth_delta = pred_depth_delta.cpu()
            pred_depth_delta = pred_depth_delta.detach().permute(0, 2, 3, 1).numpy().reshape(-1, 480, 640)

            depth_delta = pred_depth_delta[0]
            depth_delta = normalize_backward(depth_delta)

            depth_delta[cap_gray_delta_mask == 0] = 0

            if True:
                depth = (empty_depth_img - depth_delta)/ 1000
                depth = cv2.GaussianBlur(depth, (5, 5), 5)
                depth[mask == 0] = 0
                cv2.imshow('depth', depth)


        cv2.waitKey(1)
        frame_count = frame_count + 1
        end_time = time.time()
        print('fps:', 1 / (end_time - initialize_time))

    cap.release()
    cv2.destroyAllWindows()