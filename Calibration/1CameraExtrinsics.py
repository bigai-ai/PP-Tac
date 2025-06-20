from calibration import CalibrationDomeCamera
import numpy as np
import os
import cv2
import json

sensor_data_root = './sensor1/'


cfg_path = os.path.join(sensor_data_root,  'calib_extrinsic/calib_dome_position.cfg')
print(cfg_path)
calib_file_root = os.path.join(sensor_data_root, 'calib_extrinsic')
ref_path =  os.path.join(sensor_data_root, 'calib_extrinsic/ref.png')


CAMERA_MATRIX =  np.array([[299.97297618394043, 0.0, 325.2448075365583], [0.0, 299.81689506705544, 264.8979069809129], [0.0, 0.0, 1.0]])
Distortion =  np.array([[-0.25971253956661766], [0.0870041618277516], [-1.4872719847160291e-05], [-0.00022889520370614243], [-0.015351918730466424]])

caliber = CalibrationDomeCamera(cfg_path, calib_file_root, ref_path, NUM_POINT_CALIB = 5, CAMERA_MATRIX_INIT=CAMERA_MATRIX, DIST_COEFFS_INIT=Distortion, ONLY_EXTRINSIC=True)

camera_matrix, dist_coeffs, rvecs, tvecs = caliber.run(80)
# sensor
# Rotation Vectors :
#  [array([[0.00659009],
#        [0.00442339],
#        [1.59145817]])]
# Translation Vectors :
#  [array([[ 0.27462893],
#        [-1.07430713],
#        [ 7.02421485]])]

calibration_results= {
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.tolist(),
    "rvecs": rvecs[0].tolist(),
    "tvecs": tvecs[0].tolist()
}

out_file = open(os.path.join(sensor_data_root, 'calibration_new.cfg'), "w")
json.dump(calibration_results, out_file)
out_file.close()