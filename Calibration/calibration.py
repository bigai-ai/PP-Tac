
import json
import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from circile_recog import circle_recog


def load_dome_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        calib_dome_position = json.load(f)
    return calib_dome_position

def load_image_name_list(calib_file_root):
    calib_image_name_list = [i for i in os.listdir(calib_file_root) if i.endswith('.png') and ('ref' not in i)]
    return calib_image_name_list

def img_name2index(img_name):
    index = []
    for i in img_name.split('.')[0].split('_'):
        # index.append(eval(i))
        index.append(int(i))
    return index


class CalibrationDomeCamera():
    def __init__(self, cfg_path, calib_file_root, ref_path, NUM_POINT=5, NUM_POINT_CALIB=4, CAMERA_MATRIX_INIT=None, DIST_COEFFS_INIT=None, ONLY_EXTRINSIC=True):
        self.cfg_path = cfg_path
        self.calib_file_root = calib_file_root

        self.calib_dome_position = load_dome_cfg(self.cfg_path)
        self.calib_image_name_list = load_image_name_list(self.calib_file_root)
        self.ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        self.img_size = (self.ref_img.shape[1], self.ref_img.shape[0])
        self.NUM_POINT = NUM_POINT
        self.NUM_POINT_CALIB = NUM_POINT_CALIB

        if CAMERA_MATRIX_INIT is None:
            self.camera_matrix = np.array([[296.7119, 0, 302.7597], [0, 296.7701, 317.9416], [0, 0, 1.]], dtype=np.float32)
            self.ONLY_EXTRINSIC = False
        else:
            self.camera_matrix = CAMERA_MATRIX_INIT
            self.ONLY_EXTRINSIC = ONLY_EXTRINSIC

        if DIST_COEFFS_INIT is None:
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32) 
        else:
            self.dist_coeffs = DIST_COEFFS_INIT


    def run(self, num_showpoints):
        peusdo_sorted_circle_centers_all = self.peusdo_sensor_circle_center()
        dome_position = self.real_dome_center(peusdo_sorted_circle_centers_all)
        sorted_circle_centers_all_new, sorted_distance_all = self.sort_by_dome_center(peusdo_sorted_circle_centers_all, dome_position)
        position_3D_all, position_2D_all = self.extract_2D_3D_position(sorted_circle_centers_all_new, NUM_POINT_CALIB = self.NUM_POINT_CALIB)
        self.position_3D_all = position_3D_all
        self.position_2D_all = position_2D_all

        camera_matrix, dist_coeffs, rvecs, tvecs= self.calculate_calib(position_3D_all, position_2D_all,SHOWPOINT=num_showpoints)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        return camera_matrix, dist_coeffs, rvecs, tvecs

    def peusdo_sensor_circle_center(self):
        peusdo_sorted_circle_centers_all = []
        self.index_3D_all = []
        for img_name in self.calib_image_name_list[:]:
            index = img_name2index(img_name)
            tac_img = cv2.imread(osp.join(self.calib_file_root,img_name), cv2.IMREAD_GRAYSCALE)
            print(img_name)
            gray_blurred, SENSOR_CIRCLE, delta_masked, delta_masked_point, circles = circle_recog(self.ref_img, tac_img, NUM_POINT=5)

            SENSOR_CIRCLE = np.asarray(SENSOR_CIRCLE, dtype=np.float32)
            circles = np.asarray(circles, dtype=np.float32)
            SENSOR_CIRCLE_CENTER = SENSOR_CIRCLE[:2]
            circle_distances = [(i, np.linalg.norm(SENSOR_CIRCLE_CENTER-i[:2] )) for i in circles[0]]
            sorted_circles = sorted(circle_distances, key=lambda x: x[1])
            sorted_circle_centers = [i[0] for i in sorted_circles]
            
            if circles.shape[1] == self.NUM_POINT:
                peusdo_sorted_circle_centers_all.append(sorted_circle_centers)
                self.index_3D_all.append(index)
                print(f"sorted_circle_centers shape is {len(sorted_circle_centers)}")

        return peusdo_sorted_circle_centers_all

    def real_dome_center(self, peusdo_sorted_circle_centers_all, FILTEROUT=2):
        peusdo_sorted_circle_centers_all = np.asarray(peusdo_sorted_circle_centers_all)
        x_coords = peusdo_sorted_circle_centers_all[:, 0, 0]
        y_coords = peusdo_sorted_circle_centers_all[:, 0, 1]

        FILTEROUT = 2  
        x_filtered = np.sort(x_coords)[FILTEROUT:-FILTEROUT] if len(x_coords) > 2 * FILTEROUT else []
        y_filtered = np.sort(y_coords)[FILTEROUT:-FILTEROUT] if len(y_coords) > 2 * FILTEROUT else []

        x_mean = np.mean(x_filtered) if x_filtered.size > 0 else float('nan')
        y_mean = np.mean(y_filtered) if y_filtered.size > 0 else float('nan')

        dome_position = np.array([x_mean, y_mean])
        return dome_position

    def sort_by_dome_center(self,peusdo_sorted_circle_centers_all, dome_position):
        # n x 5 x 3
        sorted_circle_centers_all_new = []
        sorted_distance_all = []
        for circles in peusdo_sorted_circle_centers_all:
            circle_distances = [(i, np.linalg.norm(dome_position-i[:2] )) for i in circles]
            sorted_circles = sorted(circle_distances, key=lambda x: x[1])
            sorted_circle_centers = [i[0] for i in sorted_circles]
            sorted_circle_centers_all_new.append(sorted_circle_centers)

            sorted_distance = [ np.linalg.norm(dome_position-i[:2] ) for i in sorted_circle_centers]
            sorted_distance_all.append(sorted_distance)

        sorted_circle_centers_all_new = np.asarray(sorted_circle_centers_all_new)
        sorted_distance_all = np.asarray(sorted_distance_all)

        return sorted_circle_centers_all_new, sorted_distance_all


    def extract_2D_3D_position(self, sorted_circle_centers_all_new, NUM_POINT_CALIB=4):

        position_3D_all = []
        for i in self.index_3D_all:
            for j in i[:NUM_POINT_CALIB]:
                position = self.calib_dome_position[str(j)]
                position_3D_all.append(position)

        position_3D_all = np.asarray(position_3D_all, dtype=np.float32)
        position_2D_all = np.asarray(sorted_circle_centers_all_new[:,:NUM_POINT_CALIB,:2].reshape(-1,2), dtype=np.float32)

        return position_3D_all, position_2D_all
    

    def calculate_calib(self, position_3D_all, position_2D_all, SHOWPOINT=0):
        object_points = [np.array(position_3D_all, dtype=np.float32)]
        image_points = [np.array(position_2D_all, dtype=np.float32)]

        if self.ONLY_EXTRINSIC:
            print('--------============Calibration Start: Extrinsic =========---------------')
            ret, rvecs, tvecs = cv2.solvePnP(object_points[0], image_points[0], self.camera_matrix, self.dist_coeffs, flags = cv2.SOLVEPNP_ITERATIVE)
            camera_matrix = self.camera_matrix
            dist_coeffs = self.dist_coeffs
            rvecs = [np.asarray(rvecs)]
            tvecs = [np.asarray(tvecs)]
        else:
            print('--------============Calibration Start: Intrinsic + Extrinsic =========---------------')
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, self.img_size, self.camera_matrix, self.dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
            print("Camera matrix : \n", camera_matrix)
            print("Distortion coefficients : \n", dist_coeffs)

        print('calibration error: ', ret)
        print("Rotation Vectors : \n", rvecs)
        print("Translation Vectors : \n", tvecs)

        if SHOWPOINT!=0:
            proj_points, _ = cv2.projectPoints(object_points[0], rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
            points1 = image_points[0][:SHOWPOINT]
            points2 = np.asarray(proj_points[:SHOWPOINT].reshape(SHOWPOINT,2), dtype=np.int32)
            x1 = points1[:,0]
            y1 = points1[:,1]

            x2 = points2[:,0]
            y2 = points2[:,1]
            
            fig, ax = plt.subplots(figsize=(self.img_size[0]/100, self.img_size[1]/100))

            ax.scatter(x1, y1, color='green', label='2D points', s=1)
            ax.scatter(x2, y2, color='red', label='3D projection points',s =1)

            ax.legend()
            ax.set_xlim(0, self.img_size[0])
            ax.set_ylim(0, self.img_size[1])

            ax.invert_yaxis()

            plt.show()

        return camera_matrix, dist_coeffs, rvecs, tvecs
    
    


