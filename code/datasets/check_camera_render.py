# import numpy as np
# import cv2
# import pyrender
# import trimesh

# def load_mesh(file_path):
#     """Load mesh from an .obj file."""
#     mesh = trimesh.load(file_path)
#     return pyrender.Mesh.from_trimesh(mesh)

# def render_image(mesh, K, w2c, H, W):
#     """Render an image from the mesh using the given camera parameters."""
#     scene = pyrender.Scene()
#     scene.add(mesh)

#     # Convert w2c to c2w
#     c2w = np.linalg.inv(w2c)

#     # Set up the camera using intrinsics K
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]
#     camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    
#     # Set up camera pose
#     camera_pose = np.eye(4)
#     camera_pose[:3, :3] = c2w[:3, :3]
#     camera_pose[:3, 3] = c2w[:3, 3]
#     scene.add(camera, pose=camera_pose)

#     # Add directional light to the scene
#     directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)  # Adjusted intensity
#     scene.add(directional_light, pose=camera_pose)

#     # Add point light to the scene
#     point_light = pyrender.PointLight(color=np.ones(3), intensity=0.2)  # Adjusted intensity
#     scene.add(point_light, pose=camera_pose)

#     # Set up the renderer
#     renderer = pyrender.OffscreenRenderer(W, H)
#     color, depth = renderer.render(scene)
    
#     return color, depth

# def main():
#     # Camera intrinsics (K)
#     K = np.array([
#         [811.9282694049824, 0.0, 256.0],
#         [0.0, 811.9282694049824, 256.0],
#         [0.0, 0.0, 1.0]
#     ])

#     # Camera extrinsics (w2c)
#     w2c = np.array([
#         [-0.9263498445887507, -7.138078495581335e-18, 0.3766642608881244, 2.380584628400063e-17],
#         [-0.35286361235654806, -0.34983312172813186, -0.8678156820527256, 2.5871587468018197e-18],
#         [0.1317696342299121, -0.9368120339438163, 0.32406785794485266, 2.3713905144570306],
#         [0.0, 0.0, 0.0, 1.0]
#     ])

#     # Image dimensions
#     H, W = 512, 512

#     # Load the mesh
#     mesh_file_path = '/cluster/home/yueshi/code/PointSDF-main/example_data/kitty/physg_kitty.obj'
#     mesh = load_mesh(mesh_file_path)
    
#     # Render the image
#     color, depth = render_image(mesh, K, w2c, H, W)

#     # Save the rendered image
#     cv2.imwrite('./rendered.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

# if __name__ == "__main__":
#     main()


import numpy as np
import pyrender
import trimesh

def load_mesh(file_path):
    """Load mesh from an .obj file."""
    mesh = trimesh.load(file_path)
    return pyrender.Mesh.from_trimesh(mesh)

def create_camera_marker():
    """Create a small pyramid to represent the camera position and orientation."""
    vertices = np.array([
        [0, 0, 0],
        [0.05, 0.05, 0.1],
        [-0.05, 0.05, 0.1],
        [-0.05, -0.05, 0.1],
        [0.05, -0.05, 0.1]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 2, 3],
        [1, 3, 4]
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return pyrender.Mesh.from_trimesh(mesh)

def visualize_scene(mesh, K, w2c):
    """Visualize the mesh and camera position in a 3D viewer."""
    scene = pyrender.Scene()
    scene.add(mesh)

    # Convert w2c to c2w
    c2w = np.linalg.inv(w2c)

    # Create and add the camera marker
    camera_marker = create_camera_marker()
    scene.add(camera_marker, pose=c2w)

    # Create a viewer to display the scene
    pyrender.Viewer(scene, use_raymond_lighting=True)

def main():
    # Camera intrinsics (K)
    K = np.array([
        [811.9282694049824, 0.0, 256.0],
        [0.0, 811.9282694049824, 256.0],
        [0.0, 0.0, 1.0]
    ])

    # Camera extrinsics (w2c)
    w2c = np.array([
        [-0.9263498445887507, -7.138078495581335e-18, 0.3766642608881244, 2.380584628400063e-17],
        [-0.35286361235654806, -0.34983312172813186, -0.8678156820527256, 2.5871587468018197e-18],
        [0.1317696342299121, -0.9368120339438163, 0.32406785794485266, 2.3713905144570306],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Load the mesh
    mesh_file_path = '/cluster/home/yueshi/code/PointSDF-main/example_data/kitty/physg_kitty.obj'
    mesh = load_mesh(mesh_file_path)
    
    # Visualize the scene
    visualize_scene(mesh, K, w2c)

if __name__ == "__main__":
    main()