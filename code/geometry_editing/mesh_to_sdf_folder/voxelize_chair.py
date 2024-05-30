import numpy as np
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere
import trimesh
import skimage, skimage.measure
import os
import pdb; pdb.set_trace()
mesh = trimesh.load('/cluster/home/yueshi/code/PointSDF-main/example_data/bear/bear.obj')
mesh = scale_to_unit_sphere(mesh)

print("Scanning...")
# cloud = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=20, scan_resolution=400)
cloud = get_surface_point_cloud(mesh,sample_point_count=50000, calculate_normals=True)
cloud.show()

os.makedirs("test", exist_ok=True)
for i, scan in enumerate(cloud.scans):
    scan.save("test/scan_{:d}.png".format(i))

print("Voxelizing...")
voxels = cloud.get_voxels(128, use_depth_buffer=True)

print("Creating a mesh using Marching Cubes...")
vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
mesh.show()