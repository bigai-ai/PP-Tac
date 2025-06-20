import os
import json
import numpy as np
import cv2
import os.path as osp
import matplotlib.pyplot as plt
from render_utils import read_obj_verts, rotate_L2R_180_ply, filter_rotate_obj2ply, save_ply, read_ply
from render_utils import reproject3D, OpencvCameraPose2OpenGLCameraPose, render_depth, MergeTwoOBJ, Rotate180

sensor_data_root = './sensor1/'

with open(os.path.join(sensor_data_root, 'calibration_new.cfg'), 'r') as f:
    cfg = json.load(f)
    print(cfg)
    f.close()

camera_matrix = np.array(cfg['camera_matrix'])
dist_coeffs = np.array(cfg['dist_coeffs'])
rvecs = [np.array(cfg['rvecs'])]
tvecs = [np.array(cfg['tvecs'])]

obj_path_root = './obj'
ref_path = os.path.join(sensor_data_root, 'ref.png')
ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

obj_path = osp.join(obj_path_root, 'ref.obj')
all_point_position = read_obj_verts(obj_path)
print(all_point_position.shape)

R = cv2.Rodrigues(rvecs[0])[0]
T = tvecs[0]
camera_pose = OpencvCameraPose2OpenGLCameraPose(R, T)
extrinsics = np.concatenate((R, T), axis=1)
points = all_point_position
points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T

points_camera = np.dot(extrinsics, points).T
points_camera_depth = points_camera[:, 2]
points_camera_2D = points_camera[:, :2] / points_camera_depth.reshape(-1, 1)

k1 = dist_coeffs[0][0]
k2 = dist_coeffs[1][0]
p1 = dist_coeffs[2][0]
p2 = dist_coeffs[3][0]
k3 = dist_coeffs[4][0]

points_camera_2D_new = []
for i in points_camera_2D:
    x= i[0]
    y= i[1]
    r = x**2 + y**2
    new_x = x * (1 + k1 * r + k2 * r**2 + k3 * r**3) + 2 * p1 * x * y + p2 * (r + 2 * x**2)
    new_y = y * (1 + k1 * r + k2 * r**2 + k3 * r**3) + 2 * p2 * x * y + p1 * (r + 2 * y**2)
    points_camera_2D_new.append([new_x, new_y])
points_camera_2D = np.asarray(points_camera_2D_new)
points_camera_2D = np.concatenate((points_camera_2D, np.ones((points_camera_2D.shape[0], 1))), axis=1).T

points_camera_2D = np.dot(camera_matrix, points_camera_2D).T

x1 = points_camera_2D[:,0]
y1 = points_camera_2D[:,1]

alpha = 0.003
img_size = (640, 480)
fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100))
ax.imshow(ref_img, cmap='gray')

ax.scatter(x1, y1, color='red', label='2D points', s=5, alpha=alpha)

ax.legend()
ax.set_xlim(0, img_size[0])
ax.set_ylim(0, img_size[1])

ax.invert_yaxis()

plt.show()

# obj_path = './obj/ref.obj'
save_depth_path = os.path.join(sensor_data_root, 'ref_depth.png')
print(save_depth_path)
render_depth(obj_path,save_depth_path, camera_matrix , camera_pose, dist_coeffs, img_size= (640, 480), show = True)

height, width = ref_img.shape[:2]
center_x, center_y = width // 2, height // 2
radius = min(width, height) // 2

cv2.namedWindow('Find Center and Radius', cv2.WINDOW_NORMAL)

while True:
    img_copy = ref_img.copy()
    cv2.circle(img_copy, (center_x, center_y), radius, (255, 255, 255), 2)
    cv2.imshow('Image with Circle', img_copy)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        center_y = max(center_y - 1, 0)
    elif key == ord('s'):
        center_y = min(center_y + 1, height)
    elif key == ord('a'):
        center_x = max(center_x - 1, 0)
    elif key == ord('d'):
        center_x = min(center_x + 1, width)

    elif key == ord('n'):
        radius = max(radius - 1, 0)
    elif key == ord('m'):
        radius += 1

    elif key == ord('q'):
        break

print(f'Center: ({center_x}, {center_y})')
print(f'Radius: {radius}')
cv2.destroyAllWindows()