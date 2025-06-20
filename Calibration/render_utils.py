import numpy as np
import os 
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path as osp
import trimesh
import pyrender

def read_obj_verts(obj_path):
    with open(obj_path, 'r') as f:
        lines = f.readlines()

    all_point_position = []
    for line in lines:
        temp = line.split(' ')
        one_point_position = []
        if temp[0] == 'v':
            one_point_position.append(float(temp[1]))
            one_point_position.append(float(temp[2]))
            one_point_position.append(float(temp[3]))
            one_point_position = np.asarray(one_point_position)
            # print(one_point_position)
            if one_point_position.shape[0] == 3:
                all_point_position.append(one_point_position)
            else:
                print('The path of wrong obj file is: ', obj_path)
                print('The wrong line of obj is: ', line)
                raise ValueError('Fail to load the .obj file and the verts')

    all_point_position = np.asarray(all_point_position)
    # print(all_point_position.shape)
    return all_point_position


def rotate_L2R_180_ply(L_points):
    # rotate the L points with Z axis 180 degree
    # L_points: N x 3
    # return: N x 3
    L_points = np.asarray(L_points)
    R_points = np.zeros(L_points.shape)
    R_points[:, 0] = -L_points[:, 0]
    R_points[:, 1] = -L_points[:, 1]
    R_points[:, 2] = L_points[:, 2]
    return R_points

def filter_rotate_obj2ply(obj_root_path, pattern_list):
    for i in pattern_list:
        obj_path = os.path.join(obj_root_path, 'L_'+str(i).zfill(2)+'.obj')
        all_point_position = read_obj_verts(obj_path)
        distance = np.linalg.norm(all_point_position[:,:2], axis=1)

        filter_points = all_point_position[distance<13.5]
        save_ply(filter_points, os.path.join(obj_root_path, 'L_filter_'+str(i).zfill(2)+'.ply'))
        filter_points_rotate = rotate_L2R_180_ply(filter_points)
        save_ply(filter_points_rotate, os.path.join(obj_root_path, 'R_filter_'+str(i).zfill(2)+'.ply'))


def save_ply(points, save_path):
    # save the 3d points in .ply
    with open(save_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % points.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for point in points:
            f.write('%f %f %f\n' % (point[0], point[1], point[2]))

def read_ply(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        all_point_position = []
        for line in lines[7:]:
            temp = line.split(' ')
            one_point_position = []
            if len(temp) == 3:
                one_point_position.append(float(temp[0]))
                one_point_position.append(float(temp[1]))
                one_point_position.append(float(temp[2]))
                one_point_position = np.asarray(one_point_position)
                all_point_position.append(one_point_position)
        all_point_position = np.asarray(all_point_position)
        f.close()
    return all_point_position


def reproject3D(L_points, camera_matrix, dist_coeffs, rvecs, tvecs):
    proj_points, _ = cv2.projectPoints(L_points, rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
    return proj_points, _



def OpencvCameraPose2OpenGLCameraPose(R, T):
    extrinsics = np.concatenate((R, T), axis=1)
    camera_pose = np.concatenate((extrinsics, np.array([[0,0,0,1]])), axis=0)
    camera_pose[[1,2],:] = -camera_pose[[1,2],:]
    camera_pose = np.linalg.inv(camera_pose)
    print(camera_pose )
    return camera_pose


def render_depth(obj_path, save_path, camera_matrix, camera_pose, dist_coeffs, img_size= (640, 480), show = False, inverse=False):
    if type(obj_path):
        m = trimesh.load(obj_path, process=False, maintain_order=True)
    else:
        m = obj_path

    if inverse == True:
        print(f"inverse the obj file {obj_path}")
        try:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=np.pi,
                direction=[0, 0, 1],
                point=[0, 0, 0]
            )
            m.apply_transform(rotation_matrix)
        except Exception as e:
            print(f"Failed to apply rotation: {e}")
            return

    mesh = pyrender.Mesh.from_trimesh(m)
    scene = pyrender.Scene()
    scene.add(mesh)
    
    camera = pyrender.IntrinsicsCamera(fx=camera_matrix[0,0], fy=camera_matrix[1,1], cx=camera_matrix[0,2], cy=camera_matrix[1,2])
    scene.add(camera, pose=camera_pose)

    r = pyrender.OffscreenRenderer(img_size[0], img_size[1])
    color, depth = r.render(scene)
    depth = cv2.undistort(depth, camera_matrix, -dist_coeffs)
    if show:
        plt.figure()
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(color)
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)
        plt.show()

    # save depth map
    # depth*10  np.uint16 .png
    depth_img = (depth*10).astype(np.uint16)
    cv2.imwrite( save_path, depth_img)
    
    try:
        depth_imgg = cv2.imread(save_path, cv2.IMREAD_UNCHANGED).astype(np.int16)
    except:
        print('!!!!! Depth map saved failed !!!!!')
        print('Save path: ',save_path)

def MergeTwoOBJ(obj_path1, obj_path2, save_path=None):
    m1 = trimesh.load(obj_path1)
    m2 = trimesh.load(obj_path2)
    m = m1 + m2

    if save_path is None:
        return m
    else:
        m.export(save_path)


def Rotate180(obj_path, save_path):
    m = trimesh.load(obj_path)
    m.vertices[:,0] = -m.vertices[:,0]
    m.vertices[:,1] = -m.vertices[:,1]
    m.export(save_path)

def circle_detection(diff_gray):
    contact_mask = (diff_gray > 2).astype(np.uint8)
    contours, _ = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    if len(sorted_areas):
        cnt = contours[areas.index(sorted_areas[-1])]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        key = -1
        print('If the detected circle is suitable, press the key "q" to continue!')
        while key != ord('q'):
            center = (int(x), int(y))
            radius = int(radius)
            circle_show = cv2.cvtColor(diff_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.circle(circle_show, center, radius, (0, 255, 0), 1)
            circle_show[int(y), int(x)] = [255, 255, 255]
            cv2.imshow('contact', circle_show.astype(np.uint8))
            key = cv2.waitKey(0)
            if key == ord('w'):
                y -= 1
            elif key == ord('s'):
                y += 1
            elif key == ord('a'):
                x -= 1
            elif key == ord('d'):
                x += 1
            elif key == ord('m'):
                radius += 1
            elif key == ord('n'):
                radius -= 1
        cv2.destroyWindow('contact')
        return center, radius
    else:
        return (0, 0), 0

def mapping_data_collection(img, ref, BallRad, pixel_per_mm):
    gray_list = []
    depth_list = []
    diff_raw = ref - img
    diff_mask = (diff_raw < 150).astype(np.uint8)
    diff = diff_raw * diff_mask
    cv2.imshow('ref', ref)
    cv2.imshow('img', img)
    cv2.imshow('diff', diff)
    center, detect_radius_p = circle_detection(diff)
    if detect_radius_p:
        x = np.linspace(0, diff.shape[0] - 1, diff.shape[0])  # [0, 479]
        y = np.linspace(0, diff.shape[1] - 1, diff.shape[1])  # [0, 639]
        xv, yv = np.meshgrid(y, x)
        xv = xv - center[0]
        yv = yv - center[1]
        rv = np.sqrt(xv ** 2 + yv ** 2)
        mask = (rv < detect_radius_p)
        temp = ((xv * mask) ** 2 + (yv * mask) ** 2) * pixel_per_mm ** 2
        height_map = (np.sqrt(BallRad ** 2 - temp) * mask - np.sqrt(
            BallRad ** 2 - (detect_radius_p * pixel_per_mm) ** 2)) * mask
        height_map[np.isnan(height_map)] = 0
        diff_gray = diff.copy()
        count = 0
        for i in range(height_map.shape[0]):
            for j in range(height_map.shape[1]):
                if height_map[i, j] > 0:
                    gray_list.append(diff_gray[i, j])
                    depth_list.append(height_map[i, j])
                    count += 1
        print('Sample points number: {}'.format(count))
        return gray_list, depth_list

def get_list(gray_list, depth_list):
    GRAY_scope = int(gray_list.max())
    GRAY_Height_list = np.zeros(GRAY_scope + 1)
    for gray_number in range(GRAY_scope + 1):
        gray_height_sum = depth_list[gray_list == gray_number].sum()
        gray_height_num = (gray_list == gray_number).sum()
        if gray_height_num:
            GRAY_Height_list[gray_number] = gray_height_sum / gray_height_num
    for gray_number in range(GRAY_scope + 1):
        if GRAY_Height_list[gray_number] == 0:
            if not gray_number:
                min_index = gray_number - 1
                max_index = gray_number + 1
                for i in range(GRAY_scope - gray_number):
                    if GRAY_Height_list[gray_number + 1 + i] != 0:
                        max_index = gray_number + 1 + i
                        break
                GRAY_Height_list[gray_number] = (GRAY_Height_list[max_index] - GRAY_Height_list[min_index]) / (
                        max_index - min_index)
    return GRAY_Height_list

def raw_image_2_height_map(img_GRAY, ref_GRAY, Pixel_to_Depth, max_index, lighting_threshold):
    diff_raw = ref_GRAY - img_GRAY - lighting_threshold
    diff_mask = (diff_raw < 50).astype(np.uint8)   # 100
    diff = diff_raw * diff_mask + lighting_threshold
    print(f"height max is {np.max(diff)}")
    print(f"height min is {np.min(diff)}")
    diff[diff > max_index] = max_index
    diff[diff < 5] = 0
    diff = cv2.GaussianBlur(diff.astype(np.float32), (3, 3), 0).astype(int)     # 7,7
    print(f"height max of diff_1 is {np.max(diff)}, height min of diff_1 is {np.min(diff)}")
    height_map = Pixel_to_Depth[diff] - \
        Pixel_to_Depth[lighting_threshold]
    height_map = cv2.GaussianBlur(height_map.astype(np.float32), (3, 3), 0)
    height_map[height_map < 0] = 0
    return height_map

def height_map_2_depth_map(height_map, contact_gray_base, depth_k):
    contact_show = np.zeros_like(height_map)
    contact_show[height_map > 0] = contact_gray_base
    depth_map = height_map * depth_k + contact_show
    depth_map = depth_map.astype(np.uint8)
    return depth_map

def get_mask(ref_img, MASK_CENTER, MASK_RADIUS):
    mask = np.zeros(ref_img.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.linalg.norm([i-MASK_CENTER[0], j-MASK_CENTER[1]]) < MASK_RADIUS:
                mask[i, j] = 1
    return mask


