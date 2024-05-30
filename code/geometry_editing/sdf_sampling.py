from mesh_to_sdf_folder.mesh_to_sdf import sample_sdf_near_surface_with_sampling_pnts
import trimesh
import pyrender
import numpy as np
import pymeshfix
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
from mesh_to_sdf_folder.mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere
edited_mesh = trimesh.load('/cluster/home/yueshi/code/PointSDF-main/example_data/shiny/toaster/toaster.ply')
# points = None
mesh = scale_to_unit_sphere(edited_mesh)
points = get_surface_point_cloud(mesh,sample_point_count=50000, calculate_normals=False)

# edited_mesh = trimesh.load('../example_data/nerf_chair/nerf_chair_mesh_fix_arap.obj')
# points = np.load('../example_data/nerf_chair/nerf_chair_remove_leg_points.npy')
#(50000,3)

### fix mesh
edited_mesh = pymeshfix.MeshFix(edited_mesh.vertices, edited_mesh.faces)
edited_mesh.repair(verbose=False,joincomp=True,remove_smallest_components=False)

### to mesh
edited_mesh = trimesh.Trimesh(edited_mesh.v, edited_mesh.f)

### fill holes
# trimesh.repair.broken_faces(edited_mesh)
flag = trimesh.repair.fill_holes(edited_mesh)

print(flag)

points, sdf = sample_sdf_near_surface_with_sampling_pnts(edited_mesh, points, number_of_points=100000)#, sign_method='depth')

colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
# I commented out the following line to avoid errors
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


print(points.shape, sdf.shape)

# np.savez('../example_data/nerf_chair/nerf_chair_mesh_fix_arap.npz', points=points, sdf=sdf)
np.savez('../example_data/shiny/toaster/toaster.npz', points=points, sdf=sdf)
