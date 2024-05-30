#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import open3d as o3d


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples2(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
   
    pnts = torch.from_numpy(npz['points'])
    sdf = torch.from_numpy(npz['sdf']).unsqueeze(1)

    # random_pos = (torch.rand(subsample) * pnts.shape[0]).long()

    # sample_pos = torch.index_select(pnts, 0, random_pos)
    # sample_sdf = torch.index_select(sdf, 0, random_pos)

    sample_pos = pnts
    sample_sdf = sdf

    samples = torch.cat([sample_pos, sample_sdf], 1)

    # print(samples.shape)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        subsample
    ):
        self.subsample = subsample

        self.data_source = data_source

        # logging.debug(
        #     "using "
        #     + str(len(self.npyfiles))
        #     + " shapes from data source "
        #     + data_source
        # )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        filename = self.data_source
        return unpack_sdf_samples2(filename, self.subsample), idx
    

class GeometryEditingColor(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_source, 
        subsample
    ):
        self.subsample = subsample

        self.data_source = data_source

        filename = self.data_source

        ### load obj
        if filename.endswith('.obj'):
            obj = o3d.io.read_triangle_mesh(filename)
            self.verts = torch.from_numpy(np.asarray(obj.vertices))
            self.colors = torch.from_numpy(np.asarray(obj.vertex_colors))
        else:
            pcd = o3d.io.read_point_cloud(filename)
            self.verts = torch.from_numpy(np.asarray(pcd.points))
            self.colors = torch.from_numpy(np.asarray(pcd.colors))


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        
        

        random_pos = (torch.rand(self.subsample) * self.verts.shape[0]).long()

        sample_verts = torch.index_select(self.verts, 0, random_pos)
        sample_colors = torch.index_select(self.colors, 0, random_pos)

        return sample_verts, sample_colors, idx


if __name__ == '__main__':
    ### load obj files

    obj = o3d.io.read_triangle_mesh('/data2/code_backup/PhySG/code/geometry_editing/data/kitty-color_arap_deform.obj')
    verts = torch.from_numpy(np.asarray(obj.vertices))
    colors = torch.from_numpy(np.asarray(obj.vertex_colors))

    subsamples = 10000

    random_pos = (torch.rand(subsamples) * verts.shape[0]).long()

    sample_verts = torch.index_select(verts, 0, random_pos)
    sample_colors = torch.index_select(colors, 0, random_pos)
    print(sample_verts.shape, sample_colors.shape)


