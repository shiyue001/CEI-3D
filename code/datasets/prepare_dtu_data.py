import os
import numpy as np
from glob import glob
from PIL import Image
import cv2
import torch
import json

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[2, 2] = -1
    flip_yz[1, 1] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def camera_intrinsics(dataset_dir):

    data_info_dict = {}
    
    all_cam = np.load(os.path.join(dataset_dir,'cameras_sphere.npz'))
    all_rgb_files = sorted(glob(os.path.join(dataset_dir, 'image/*')))
    length = len(all_rgb_files)

    fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
    for i in range(length):
        img_path = all_rgb_files[i]
        image = Image.open(img_path)
        W, H = image.size

        img_name = os.path.split(img_path)[-1]
        # idx = img_name[:-4]
        # import pdb; pdb.set_trace()
        P = all_cam["world_mat_" + str(i)] 
        P = P[:3]

        K, R, t = cv2.decomposeProjectionMatrix(P)[:3] #R is the rotation matrix of w2c
        K = K / K[2,2]

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose() 
        pose[:3, 3] = (t[:3] / t[3])[:, 0] 
 
        #这部分是否冗余，目标就是得到w2c，为什么先从w2c得到c2w再转回w2c呢？是否多此一举
        #之所以这样做，是为了用convert_pose转成opencv类型的数据，猜测physg用的是opencv类型的
        #pose: W2C
        pose = np.linalg.inv(pose) # C2W
        pose = convert_pose(pose)
        pose = np.linalg.inv(pose) # W2C


        # 来进行normalize的
        scale_mtx = all_cam.get("scale_mat_" + str(i))
        if scale_mtx is not None:
            norm_trans = scale_mtx[:3, 3:]
            norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]
            pose[:3, 3:] -= norm_trans
            pose[:3, 3:] /= norm_scale

        fx += torch.tensor(K[0, 0]) * 1.0
        fy += torch.tensor(K[1, 1]) * 1.0
        cx += (torch.tensor(K[0, 2]) + 0.0) * 1.0
        cy += (torch.tensor(K[1, 2]) + 0.0) * 1.0
        


        W2C = [float(i) for i in list(pose.flatten())]
        data_info_dict.update({
            img_name:{
                "K":K,
                "W2C":W2C,
                "img_size":[W,H]
            }
        })

    fx /= length
    fy /= length
    cx /= length
    cy /= length

    intrinsics = np.array([
        [fx, 0., cx, 0], 
        [0., fy, cy, 0],
        [0., 0, 1, 0],
        [0, 0, 0, 1]
    ])

    K = [float(i) for i in list(intrinsics.flatten())]

    for key, val in data_info_dict.items():
        val['K'] = K

    with open(os.path.join(dataset_dir, 'cam_dict_train_convert_norm.json'), 'w') as json_f:
        json.dump(data_info_dict, json_f, indent=4)
        
if __name__ == "__main__":
    dataset_dir = '/cluster/home/yueshi/code/PointSDF-main/example_data/bear/train'
    camera_intrinsics(dataset_dir)