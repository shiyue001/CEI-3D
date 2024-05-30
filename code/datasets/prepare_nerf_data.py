import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import json
import copy
import open3d as o3d

def get_tf_cams(cam_dict, target_radius=1.):
    cam_centers = []
    for im_name in cam_dict:
        W2C = np.array(cam_dict[im_name]['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    # points = np.array(cam_centers, dtype = 'float64').squeeze()
    # print(points.shape)
    # x, y, z, r = sphere_surface(points)
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1 # *->/
    translate = -center
    # translate = [-x, -y, -z]
    scale = target_radius / radius
    return translate, scale


def normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1., in_geometry_file=None, out_geometry_file=None):
    with open(in_cam_dict_file) as fp:
        in_cam_dict = json.load(fp)

    translate, scale = get_tf_cams(in_cam_dict, target_radius=target_radius)

    if in_geometry_file is not None and out_geometry_file is not None:
        # check this page if you encounter issue in file io: http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
        geometry = o3d.io.read_triangle_mesh(in_geometry_file)
        tf_translate = np.eye(4)
        tf_translate[:3, 3:4] = translate
        tf_scale = np.eye(4)
        tf_scale[:3, :3] *= scale
        tf = np.matmul(tf_scale, tf_translate)
        geometry_norm = geometry.transform(tf)
        o3d.io.write_triangle_mesh(out_geometry_file, geometry_norm)
    def transform_pose(W2C, translate, scale):
        C2W = np.linalg.inv(W2C)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        # cam_center = (cam_center ) * scale

        C2W[:3, 3] = cam_center
        # C2W[1, 3] -= 1.1
        # C2W[2, 3] -= 1.1

        return np.linalg.inv(C2W)

    out_cam_dict = copy.deepcopy(in_cam_dict)
    for img_name in out_cam_dict:
        W2C = np.array(out_cam_dict[img_name]['W2C']).reshape((4, 4))
        W2C = transform_pose(W2C, translate, scale)

        assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
        out_cam_dict[img_name]['W2C'] = list(W2C.flatten())

    with open(out_cam_dict_file, 'w') as fp:
        json.dump(out_cam_dict, fp, indent=2, sort_keys=True)


def sphere_surface(points):
    num_points = points.shape[0]
    # print(num_points)
    x, y, z= points[:, 0], points[:, 1], points[:, 2]
    x_avr, y_avr, z_avr = sum(x)/num_points, sum(y)/num_points, sum(z)/num_points  
    xx_avr, yy_avr, zz_avr = sum(x*x)/num_points, sum(y*y)/num_points, sum(z*z)/num_points
    xy_avr, xz_avr, yz_avr = sum(x*y)/num_points, sum(x*z)/num_points, sum(y*z)/num_points
    
    xxx_avr = sum(x * x * x) / num_points
    xxy_avr = sum(x * x * y) / num_points
    xxz_avr = sum(x * x * z) / num_points
    xyy_avr = sum(x * y * y) / num_points
    xzz_avr = sum(x * z * z) / num_points
    yyy_avr = sum(y * y * y) / num_points
    yyz_avr = sum(y * y * z) / num_points
    yzz_avr = sum(y * z * z) / num_points
    zzz_avr = sum(z * z * z) / num_points
    
    A = np.array([[xx_avr - x_avr * x_avr, xy_avr - x_avr * y_avr, xz_avr - x_avr * z_avr],
                  [xy_avr - x_avr * y_avr, yy_avr - y_avr * y_avr, yz_avr - y_avr * z_avr],
                  [xz_avr - x_avr * z_avr, yz_avr - y_avr * z_avr, zz_avr - z_avr * z_avr]])
    b = np.array([xxx_avr - x_avr * xx_avr + xyy_avr - x_avr * yy_avr + xzz_avr - x_avr * zz_avr,
                  xxy_avr - y_avr * xx_avr + yyy_avr - y_avr * yy_avr + yzz_avr - y_avr * zz_avr,
                  xxz_avr - z_avr * xx_avr + yyz_avr - z_avr * yy_avr + zzz_avr - z_avr * zz_avr])
    b = b / 2
    center = np.linalg.solve(A, b)
    x0, y0, z0= center[0], center[1], center[2]
    r2 = xx_avr - 2 * x0 * x_avr + x0 * x0 + yy_avr - 2 * y0 * y_avr + y0 * y0 + zz_avr - 2 * z0 * z_avr + z0 * z0
    r = r2 ** 0.5
    return center[0], center[1], center[2], r*2

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

scan_id = 'ball'
dataset_dir = '../../example_data/shiny/%s' % scan_id
split = 'test'
json_file  = os.path.join(dataset_dir, 'transforms_%s.json' % split)
mask_save_dir = os.path.join(dataset_dir, split, 'mask')

if not os.path.isdir(mask_save_dir):
    os.makedirs(mask_save_dir)
    
with open(json_file, 'r') as fp:
    info = json.load(fp)
    camera_angle_x = float(info['camera_angle_x'])
    frames_info = info['frames']
    
data_info_dict = {}

for frame_info in frames_info:
    img_path = frame_info['file_path']
    img_name = os.path.split(img_path)[-1]
    img = Image.open(os.path.join(dataset_dir, split, 'image', '%s.png' % img_name))
    H, W = img.size

    # mask = np.array(img)[:,:,-1]>=200
    #yue:save the mask
    # plt.imsave(os.path.join(mask_save_dir, '%s.png' % img_name), mask.astype(np.uint8)*255, cmap='gray')

    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.eye(4, dtype=np.float32)
    K[0,0], K[1,1] = focal, focal
    K[0,2], K[1,2] = H/2, W/2

    rotation = frame_info['rotation']
    transform_matrix = frame_info['transform_matrix']
    transform_matrix = np.array(transform_matrix)
    C2W = convert_pose(transform_matrix)
    # transform_matrix[2, 3] *= -1
    W2C = np.linalg.inv(C2W)
    # W2C[2, 3] *= -1
    K = [float(i) for i in K.flatten()]
    W2C = [float(i) for i in W2C.flatten()]

    data_info_dict.update({
        '%s.png' % img_name:{
            "K":K,
            "W2C":W2C,
            "img_size":[H, W]
        }
    })

with open(os.path.join(dataset_dir, split, 'cam_dict.json'), 'w') as json_f:
    json.dump(data_info_dict, json_f, indent=4)
    
in_cam_dict_file = os.path.join(dataset_dir, split, 'cam_dict.json')
out_cam_dict_file = os.path.join(dataset_dir, split, 'cam_dict_norm.json')
normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1.)