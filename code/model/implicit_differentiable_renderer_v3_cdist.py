import code
from xml.dom import INDEX_SIZE_ERR
from bisect import bisect_left, bisect_right
from tqdm import trange

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import rend_util
from utils.rend_util import quat_to_rot, lift

from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from model.sg_envmap_material_v2 import Diffuse_albedo_layers, EnvmapMaterialNetwork_v2
from model.sg_render import render_with_sg


### modified by three at 2022.10.16
### get edited points list, first project 2D edited points back to 3D points, 
### traced back according to the view_dir and location, get the 2D coordinate at the current rendering view


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        self.feature_vector_size = feature_vector_size
        # print('ImplicitNetowork feature_vector_size: ', self.feature_vector_size)
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires_xyz=0
    ):
        super().__init__()

        self.feature_vector_size = feature_vector_size
        # print('RenderingNetowork feature_vector_size: ', self.feature_vector_size)

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            # print('Applying positional encoding to view directions: ', multires_view)
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.embedxyz_fn = None
        if multires_xyz > 0:
            # print('Applying positional encoding to xyz: ', multires_xyz)
            embedxyz_fn, input_ch = get_embedder(multires_xyz)
            self.embedxyz_fn = embedxyz_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.embedxyz_fn is not None:
            points = self.embedxyz_fn(points)

        if feature_vectors is not None:
            if self.mode == 'idr':
                rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
            elif self.mode == 'no_view_dir':
                rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
            elif self.mode == 'no_normal':
                rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:
            if self.mode == 'idr':
                rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
            elif self.mode == 'no_view_dir':
                rendering_input = torch.cat([points, normals], dim=-1)
            elif self.mode == 'no_normal':
                rendering_input = torch.cat([points, view_dirs], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return (x + 1.) / 2.


class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.envmap_material_network = EnvmapMaterialNetwork_v2(**conf.get_config('envmap_material_network'))
        self.diffuse_albedo_layers = Diffuse_albedo_layers(**conf.get_config('diffuse_albedo_layers'))
        self.diffuse_albedo_layers_finetuned = Diffuse_albedo_layers(**conf.get_config('diffuse_albedo_layers'))

        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def freeze_geometry(self):
        for param in self.implicit_network.parameters():
            param.requires_grad = False

    def unfreeze_geometry(self):
        for param in self.implicit_network.parameters():
            param.requires_grad = True

    def freeze_idr(self):
        self.freeze_geometry()
        for param in self.rendering_network.parameters():
            param.requires_grad = False

    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)
        diffuse_rgb = input['diffuse_rgb']

        # print('model input size', diffuse_rgb.shape, object_mask.shape)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # print('ray_dir: ', ray_dirs.shape, 'camera_loc: ', cam_loc.shape)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)

        self.implicit_network.train()

        # print('network object mask shape: ', network_object_mask.shape)

        # print('points shape', points.shape)

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3) # 坐标转换

        # print('points shape: ', points.shape)

        sdf_output = self.implicit_network(points)[:, 0:1] # 通过MLP输出物体几何
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask] # 物体表面点
            diffuse_rgb = diffuse_rgb[surface_mask] if diffuse_rgb is not None else None
            grad_theta = None

        # print(differentiable_surface_points.shape)

        idr_rgb_values = torch.ones_like(points).float().cuda()
        sg_rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.ones_like(points).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        sg_diffuse_albedo_values = torch.ones_like(points).float().cuda() # diffuse_albedo初始化为1
        sg_specular_rgb_values = torch.zeros_like(points).float().cuda() # specular_rgb初始化为0
        if differentiable_surface_points.shape[0] > 0:

            # print('differentiable_surface_points: ', differentiable_surface_points.shape)
            # differentiable_surface_points[:1000, 0] = 0.4
            # differentiable_surface_points[:1000, 1] = 0
            # differentiable_surface_points[:1000, 2] = 0

            view_dirs = -ray_dirs[surface_mask]  # ----> camera
            ret = self.get_rbg_value(differentiable_surface_points, view_dirs, diffuse_rgb)

            idr_rgb_values[surface_mask] = ret['idr_rgb']
            sg_rgb_values[surface_mask] = ret['sg_rgb']
            normal_values[surface_mask] = ret['normals']

            sg_diffuse_rgb_values[surface_mask] = ret['sg_diffuse_rgb']
            sg_diffuse_albedo_values[surface_mask] = ret['sg_diffuse_albedo']
            sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb']

        output = {
            'points': points,
            'idr_rgb_values': idr_rgb_values,
            'sg_rgb_values': sg_rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'sg_diffuse_rgb_values': sg_diffuse_rgb_values,
            'sg_diffuse_albedo_values': sg_diffuse_albedo_values,
            'sg_specular_rgb_values': sg_specular_rgb_values,
            'differentiable_surface_points': differentiable_surface_points,
        }

        return output

    def get_rbg_value(self, points, view_dirs, diffuse_rgb=None):
        feature_vectors = None
        if self.feature_vector_size > 0:
            output = self.implicit_network(points)
            feature_vectors = output[:, 1:]

        g = self.implicit_network.gradient(points) #计算梯度得到法线
        normals = g[:, 0, :]
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        ret = { 'normals': normals, }

        ### idr renderer
        idr_rgb = self.rendering_network(points, normals, view_dirs, feature_vectors)
        ret['idr_rgb'] = idr_rgb

        ### sg renderer
        sg_envmap_material = self.envmap_material_network(points)

        diffuse_albedo = self.diffuse_albedo_layers(points)
        sg_envmap_material.update({'sg_diffuse_albedo': diffuse_albedo})

        sg_ret = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                roughness=sg_envmap_material['sg_roughness'],
                                diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                normal=normals, viewdirs=view_dirs,
                                blending_weights=sg_envmap_material['sg_blending_weights'],
                                diffuse_rgb=diffuse_rgb)
        ret.update(sg_ret)
        return ret

    def render_sg_rgb(self, mask, normals, view_dirs, diffuse_albedo):
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        ### sg renderer
        sg_envmap_material = self.envmap_material_network(points=None)
        ### split
        split_size = 20000
        normals_split = torch.split(normals, split_size, dim=0)
        view_dirs_split = torch.split(view_dirs, split_size, dim=0)
        diffuse_albedo_split = torch.split(diffuse_albedo, split_size, dim=0)
        merged_ret = {}
        for i in range(len(normals_split)):
            sg_ret = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                    specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                    roughness=sg_envmap_material['sg_roughness'],
                                    diffuse_albedo=diffuse_albedo_split[i],
                                    normal=normals_split[i], viewdirs=view_dirs_split[i],
                                    blending_weights=sg_envmap_material['sg_blending_weights'])
            if i == 0:
                for x in sorted(sg_ret.keys()):
                    merged_ret[x] = [sg_ret[x].detach(), ]
            else:
                for x in sorted(sg_ret.keys()):
                    merged_ret[x].append(sg_ret[x].detach())
        for x in sorted(merged_ret.keys()):
            merged_ret[x] = torch.cat(merged_ret[x], dim=0)

        sg_ret = merged_ret
        ### maskout
        for x in sorted(sg_ret.keys()):
            sg_ret[x][~mask] = 1.

        output = {
            'sg_rgb_values': sg_ret['sg_rgb'],
            'sg_diffuse_rgb_values': sg_ret['sg_diffuse_rgb'],
            'sg_diffuse_albedo_values': diffuse_albedo,
            'sg_specular_rgb_values': sg_ret['sg_specular_rgb'],
        }

        return output


    def get_surface_points(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)
    
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3) # 坐标转换

        ray_dirs = ray_dirs.reshape(-1, 3)

        surface_mask = network_object_mask
        differentiable_surface_points = points[surface_mask]

        output = {
            'points': points,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'differentiable_surface_points': differentiable_surface_points,
        }

        return output


    def get_edited_3d_points(self, input, edited_2d_positions):

        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        if pose.shape[1] == 7: #In case of quaternion vector representation
            cam_loc = pose[:, 4:]
            R = quat_to_rot(pose[:,:4])
            p = torch.eye(4).repeat(pose.shape[0],1,1).float()
            p[:, :3, :3] = R
            p[:, :3, 3] = cam_loc
        else: # In case of pose matrix representationutils
            cam_loc = pose[:, :3, 3]
            p = pose

        batch_size, num_samples = edited_2d_positions.shape[0], edited_2d_positions.shape[1]


        depth = torch.ones((batch_size, num_samples)).cuda()
        x_cam = edited_2d_positions[:, :, 0].view(batch_size, -1)
        y_cam = edited_2d_positions[:, :, 1].view(batch_size, -1)
        z_cam = depth.view(batch_size, -1)

        pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

        # permute for batch matrix product
        pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

        world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
        ray_dirs = world_coords - cam_loc[:, None, :]
        ray_dirs_norm = F.normalize(ray_dirs, dim=2)

        batch_size, num_pixels, _ = ray_dirs_norm.shape
        object_mask = torch.ones((num_pixels, )).bool().cuda()
        
        # print(batch_size, num_pixels, ray_dirs_norm.shape)


        self.implicit_network.eval()

        points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                cam_loc=cam_loc,
                                                object_mask=object_mask,
                                                ray_directions=ray_dirs_norm)

        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs_norm).reshape(-1, 3)

        return points


    def get_edited_pixel_under_current_viewdir(self, input, edited_3d_positions, threshold=1e-9):

        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)


        if pose.shape[1] == 7: #In case of quaternion vector representation
            cam_loc = pose[:, 4:]
            R = quat_to_rot(pose[:,:4])
            p = torch.eye(4).repeat(pose.shape[0],1,1).float()
            p[:, :3, :3] = R
            p[:, :3, 3] = cam_loc
        else: # In case of pose matrix representationutils
            cam_loc = pose[:, :3, 3]
            p = pose

        batch_size, num_samples, _ = uv.shape

        depth = torch.ones((batch_size, num_samples)).cuda()
        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = depth.view(batch_size, -1)

        pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

        # permute for batch matrix product
        pixel_points_cam = pixel_points_cam.permute(0, 2, 1)


        ###########################################################################################
        ### for edited 3d points
        ### directions of camera and edited 3d points
        # print('edited 3d positions shape: ', edited_3d_positions.shape)
        ray_dirs = edited_3d_positions - cam_loc[:, None, :]
        ray_dirs_norm = F.normalize(ray_dirs, dim=2)


        batch_size, num_pixels, _ = ray_dirs_norm.shape
        object_mask = torch.ones((num_pixels, )).bool().cuda()

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                cam_loc=cam_loc,
                                                object_mask=object_mask,
                                                ray_directions=ray_dirs_norm)

        self.implicit_network.train()

        ### points on the sdf surface, viewing from the camera with direction of ray_dir
        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs_norm).reshape(-1, 3)

        
        #############################################################################################
        ### TODO: determine whether the `points` here is the `edited_3d_points`
        ### `edited_ray_dirs` are the rays in `ray_dirs_norm`

        # print(threshold)

        dist = torch.sum((edited_3d_positions-points)**2, dim=-1).squeeze(0)
        dist = dist.detach().cpu().numpy()

        # print(dist, min(dist), max(dist))
        edited_idx = np.argwhere(dist<threshold)
        edited_idx = torch.from_numpy(edited_idx).squeeze().cuda()

        # print(edited_idx, edited_idx.shape)

        # print(edited_idx, edited_idx.shape)

        edited_ray_dirs = torch.index_select(ray_dirs_norm, dim=1, index=edited_idx)

        #############################################################################################
        ### get edited 2d coords according to the edited_ray_dirs
        ### there is something wrong in this part

        world_coords = edited_ray_dirs + cam_loc[:, None, :]
        a, b, c = world_coords.shape
        new_world_coords = torch.ones(a,b,c+1)
        new_world_coords[:,:,:3] = world_coords
        new_world_coords = new_world_coords.permute(0, 2, 1).cuda()

        edited_pixels = torch.bmm(torch.inverse(p), new_world_coords).squeeze(0)

        # print('edited_pixels: ', edited_pixels.shape)


        edited_pixels_coords = torch.tensor([]).cuda()
        
        ### TODO: vectorize to speed up, ~6min with a loop
        for idx in trange(edited_pixels.shape[1]):
            x, y, z = edited_pixels[:3, idx]
            x_norm = x / z
            y_norm = y / z
            edited_pixel_coord = self.get_pixel_coords(x_norm, y_norm, pixel_points_cam)
            edited_pixel_coord = torch.tensor(edited_pixel_coord).cuda()
            edited_pixels_coords = torch.cat((edited_pixels_coords, edited_pixel_coord), dim=0)

        return edited_pixels_coords



    def forward_dual_mlp(self, input, edited_3d_positions, threshold=1e-9, print_output=True, assign_roughness=None):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)


        if pose.shape[1] == 7: #In case of quaternion vector representation
            cam_loc = pose[:, 4:]
            R = quat_to_rot(pose[:,:4])
            p = torch.eye(4).repeat(pose.shape[0],1,1).float()
            p[:, :3, :3] = R
            p[:, :3, 3] = cam_loc
        else: # In case of pose matrix representationutils
            cam_loc = pose[:, :3, 3]
            p = pose

        ##########################################################################################
        ### get 2d pixel coordinates under current camera location and direction

        batch_size, num_samples, _ = uv.shape

        depth = torch.ones((batch_size, num_samples)).cuda()
        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = depth.view(batch_size, -1)

        pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

        # permute for batch matrix product
        pixel_points_cam = pixel_points_cam.permute(0, 2, 1)


        #############################################################################################
        ### all points
        
        world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
        ray_dirs = world_coords - cam_loc[:, None, :]
        ray_dirs_norm = F.normalize(ray_dirs, dim=2)
        
        batch_size, num_pixels, _ = ray_dirs_norm.shape
        object_mask = torch.ones((num_pixels, )).bool().cuda()

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs_norm)
        
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs_norm).reshape(-1, 3)


        sdf_output = self.implicit_network(points)[:, 0:1] # 通过MLP输出物体几何
        ray_dirs = ray_dirs_norm.reshape(-1, 3)

        surface_mask = network_object_mask
        differentiable_surface_points = points[surface_mask] # 物体表面点
        grad_theta = None

        view_dirs = -ray_dirs[surface_mask]  # ----> camera

        ###############################################################################################
        ### split edited and unedited points

        if print_output:
            print('points', points.shape, 'edited_3d_positions: ', edited_3d_positions.shape)

        distance = torch.cdist(points, edited_3d_positions.squeeze(0))
        vals, idxs = torch.min(distance, 1)

        if print_output:
            print('vals.shape', vals.shape)
            print('threshold: ', threshold)

        edited_idxs = vals < threshold
        unedited_idxs = ~edited_idxs

        if print_output:
            print('unedited_idx: ', unedited_idxs)

        if print_output:
            print('index.shape', (edited_idxs*surface_mask).shape)


        edited_view_dirs = -ray_dirs[edited_idxs*surface_mask]
        unedited_view_dirs = -ray_dirs[unedited_idxs*surface_mask]

        edited_differentiable_surface_points = points[edited_idxs*surface_mask]
        unedited_differentiable_surface_points = points[unedited_idxs*surface_mask]

        unedited_pixels_idx = np.argwhere((unedited_idxs*surface_mask).cpu().numpy()==True).squeeze(-1)
        unedited_pixels_idx = torch.tensor(unedited_pixels_idx).cuda()

        edited_pixels_idx = np.argwhere((edited_idxs*surface_mask).cpu().numpy()==True).squeeze(-1)
        edited_pixels_idx = torch.tensor(edited_pixels_idx).cuda()


        ###############################################################################################


        ###
        differentiable_surface_points_dict = {'unedited':unedited_differentiable_surface_points, 'edited':edited_differentiable_surface_points, 'combined':differentiable_surface_points}
        view_dirs_dict = {'unedited':unedited_view_dirs, 'edited':edited_view_dirs, 'combined':view_dirs}
        split_idx = {'unedited':unedited_pixels_idx, 'edited':edited_pixels_idx}

        idr_rgb_values = torch.ones_like(points).float().cuda()
        sg_rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.ones_like(points).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        sg_diffuse_albedo_values = torch.ones_like(points).float().cuda() # diffuse_albedo初始化为1
        sg_specular_rgb_values = torch.zeros_like(points).float().cuda() # specular_rgb初始化为0
        if differentiable_surface_points.shape[0] > 0:

            ret = self.get_rbg_value_dual_mlp(differentiable_surface_points_dict, view_dirs_dict, split_idx, assign_roughness=assign_roughness)

            idr_rgb_values[surface_mask] = ret['idr_rgb']
            sg_rgb_values[surface_mask] = ret['sg_rgb']
            normal_values[surface_mask] = ret['normals']

            sg_diffuse_rgb_values[surface_mask] = ret['sg_diffuse_rgb']
            sg_diffuse_albedo_values[surface_mask] = ret['sg_diffuse_albedo']
            sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb']

        output = {
            'points': points,
            'idr_rgb_values': idr_rgb_values,
            'sg_rgb_values': sg_rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'sg_diffuse_rgb_values': sg_diffuse_rgb_values,
            'sg_diffuse_albedo_values': sg_diffuse_albedo_values,
            'sg_specular_rgb_values': sg_specular_rgb_values,
            'differentiable_surface_points': differentiable_surface_points,
        }

        return output
        

    def get_rbg_value_dual_mlp(self, differentiable_surface_points_dict, view_dirs_dict, split_idx, diffuse_rgb=None, assign_roughness=None):

        points = differentiable_surface_points_dict['combined']
        unedited_points = differentiable_surface_points_dict['unedited']
        edited_points = differentiable_surface_points_dict['edited']

        view_dirs = view_dirs_dict['combined']


        feature_vectors = None
        if self.feature_vector_size > 0:
            output = self.implicit_network(points)
            feature_vectors = output[:, 1:]

        g = self.implicit_network.gradient(points) #计算梯度得到法线
        normals = g[:, 0, :]
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        ret = { 'normals': normals, }

        ### idr renderer
        idr_rgb = self.rendering_network(points, normals, view_dirs, feature_vectors)
        ret['idr_rgb'] = idr_rgb

        ### sg renderer
        sg_envmap_material = self.envmap_material_network(points)


        ### split points into edited and unedited
        ### edited points are fed into diffuse_albedo_layer_finetuned
        ### unedited points are fed into diffuse_albedo_layer

        unedited_diffuse_albedo = self.diffuse_albedo_layers(unedited_points)
        # print('unedited: ', unedited_diffuse_albedo.shape, unedited_points.shape)

        edited_diffuse_albedo = self.diffuse_albedo_layers_finetuned(edited_points)
        # print('edited: ', edited_diffuse_albedo.shape, edited_points.shape)

        diffuse_albedo = self.merge_diffuse_albedo(unedited_diffuse_albedo, edited_diffuse_albedo, split_idx)


        sg_envmap_material.update({'sg_diffuse_albedo': diffuse_albedo})


        print('roughness: ', sg_envmap_material['sg_roughness'])

        roughness = sg_envmap_material['sg_roughness']

        if assign_roughness is not None:
            roughness[0, 0] = assign_roughness
            print(roughness)
            print('roughness modified!')


        sg_ret = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                roughness=sg_envmap_material['sg_roughness'],
                                diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                normal=normals, viewdirs=view_dirs,
                                blending_weights=sg_envmap_material['sg_blending_weights'],
                                diffuse_rgb=diffuse_rgb)
        ret.update(sg_ret)
        return ret


    def get_pixel_coords(self, x, y, pixels):

        # print(pixels.shape) ### 1*4*(H*W)

        # print(np.unique(pixels[0].cpu().numpy().astype(np.float16)))
        # exit()
        x_vals = list(np.unique(pixels[0, 0, :].cpu().numpy().astype(np.float16)))
        y_vals = list(np.unique(pixels[0, 1, :].cpu().numpy().astype(np.float16)))
        # print(vals, len(vals))

        # print(len(x_vals), len(y_vals))
        x_coor = bisect_left(x_vals, x)
        y_coor = bisect_left(y_vals, y)

        # print(x_coor, y_coor)

        return np.array([[x_coor-1, y_coor-1], [x_coor, y_coor-1], [x_coor-1, y_coor], [x_coor, y_coor]]).astype(np.float32)


    def merge_diffuse_albedo(self, unedited_diffuse_albedo, edited_diffuse_albedo, split_idx):

        length = unedited_diffuse_albedo.shape[0] + edited_diffuse_albedo.shape[0]
        edited_idx = split_idx['edited']
        unedited_idx = split_idx['unedited']

        # print(edited_idx, unedited_idx)

        ### re-sort, since we only take the mask part
        tmp = torch.cat((edited_idx, unedited_idx), dim=0)
        _, idx = tmp.sort(dim=0)
        _, rank = idx.sort(dim=0)

        edited_idx = rank[:edited_idx.shape[0]]
        unedited_idx = rank[edited_idx.shape[0]:]

        # print('length: ', length)
        # print(edited_idx.shape)
        # print(unedited_idx.shape)

        diffuse_albedo = torch.zeros((length, 3)).cuda()
        diffuse_albedo[edited_idx] = edited_diffuse_albedo
        diffuse_albedo[unedited_idx] = unedited_diffuse_albedo

        return diffuse_albedo


    def forward_single_mlp(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)


        if pose.shape[1] == 7: #In case of quaternion vector representation
            cam_loc = pose[:, 4:]
            R = quat_to_rot(pose[:,:4])
            p = torch.eye(4).repeat(pose.shape[0],1,1).float()
            p[:, :3, :3] = R
            p[:, :3, 3] = cam_loc
        else: # In case of pose matrix representationutils
            cam_loc = pose[:, :3, 3]
            p = pose

        ##########################################################################################
        ### get 2d pixel coordinates under current camera location and direction

        batch_size, num_samples, _ = uv.shape

        depth = torch.ones((batch_size, num_samples)).cuda()
        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = depth.view(batch_size, -1)

        pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

        # permute for batch matrix product
        pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

        world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
        ray_dirs = world_coords - cam_loc[:, None, :]
        ray_dirs_norm = F.normalize(ray_dirs, dim=2)
        
        batch_size, num_pixels, _ = ray_dirs_norm.shape
        object_mask = torch.ones((num_pixels, )).bool().cuda()

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs_norm)
        
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs_norm).reshape(-1, 3)

        ###############################################################################################


        sdf_output = self.implicit_network(points)[:, 0:1] # 通过MLP输出物体几何
        ray_dirs = ray_dirs_norm.reshape(-1, 3)

        surface_mask = network_object_mask
        differentiable_surface_points = points[surface_mask] # 物体表面点
        grad_theta = None

        view_dirs = -ray_dirs[surface_mask]  # ----> camera


        idr_rgb_values = torch.ones_like(points).float().cuda()
        sg_rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.ones_like(points).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        sg_diffuse_albedo_values = torch.ones_like(points).float().cuda() # diffuse_albedo初始化为1
        sg_specular_rgb_values = torch.zeros_like(points).float().cuda() # specular_rgb初始化为0

        if differentiable_surface_points.shape[0] > 0:

            ret = self.get_rbg_value_single_mlp(differentiable_surface_points, view_dirs)

            idr_rgb_values[surface_mask] = ret['idr_rgb']
            sg_rgb_values[surface_mask] = ret['sg_rgb']
            normal_values[surface_mask] = ret['normals']

            sg_diffuse_rgb_values[surface_mask] = ret['sg_diffuse_rgb']
            sg_diffuse_albedo_values[surface_mask] = ret['sg_diffuse_albedo']
            sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb']

        output = {
            'points': points,
            'idr_rgb_values': idr_rgb_values,
            'sg_rgb_values': sg_rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'sg_diffuse_rgb_values': sg_diffuse_rgb_values,
            'sg_diffuse_albedo_values': sg_diffuse_albedo_values,
            'sg_specular_rgb_values': sg_specular_rgb_values,
            'differentiable_surface_points': differentiable_surface_points,
        }

        return output
        

    def get_rbg_value_single_mlp(self, differentiable_surface_points, view_dirs, flag='finetuned', diffuse_rgb=None):

        feature_vectors = None
        if self.feature_vector_size > 0:
            output = self.implicit_network(differentiable_surface_points)
            feature_vectors = output[:, 1:]

        g = self.implicit_network.gradient(differentiable_surface_points)
        normals = g[:, 0, :]
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        ret = { 'normals': normals, }

        ### idr renderer
        idr_rgb = self.rendering_network(differentiable_surface_points, normals, view_dirs, feature_vectors)
        ret['idr_rgb'] = idr_rgb

        ### sg renderer
        sg_envmap_material = self.envmap_material_network(differentiable_surface_points)

        if flag == 'finetuned':
            diffuse_albedo = self.diffuse_albedo_layers_finetuned(differentiable_surface_points)
        else:
            diffuse_albedo = self.diffuse_albedo_layers(differentiable_surface_points)

        # print('unedited: ', diffuse_albedo.shape, differentiable_surface_points.shape)


        sg_envmap_material.update({'sg_diffuse_albedo': diffuse_albedo})


        sg_ret = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                roughness=sg_envmap_material['sg_roughness'],
                                diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                normal=normals, viewdirs=view_dirs,
                                blending_weights=sg_envmap_material['sg_blending_weights'],
                                diffuse_rgb=diffuse_rgb)
        ret.update(sg_ret)
        return ret


    def forward_geometry_editing(self, input, edited_pixels_coords):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)


        if pose.shape[1] == 7: #In case of quaternion vector representation
            cam_loc = pose[:, 4:]
            R = quat_to_rot(pose[:,:4])
            p = torch.eye(4).repeat(pose.shape[0],1,1).float()
            p[:, :3, :3] = R
            p[:, :3, 3] = cam_loc
        else: # In case of pose matrix representationutils
            cam_loc = pose[:, :3, 3]
            p = pose

        ##########################################################################################
        ### get 2d pixel coordinates under current camera location and direction

        batch_size, num_samples_0, _ = uv.shape

        depth = torch.ones((batch_size, num_samples_0)).cuda()
        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = depth.view(batch_size, -1)

        pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

        # permute for batch matrix product
        pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

        #############################################################################################
        ### get difference of 2d pixel coords and edited pixel coords

        # print('edited_pixels_coords: ', edited_pixels_coords.shape)
        uv_0 = uv.squeeze().squeeze()

        # print('uv_0: ', uv_0)

        # print('uv:', uv_0.shape, edited_pixels_coords.shape)

        if edited_pixels_coords.shape[0] == 0:
            m = torch.tensor([False, ]*uv_0.shape[0]).bool().cuda()
        else:
            m = (uv_0[:, None] == edited_pixels_coords).all(-1).any(1)


        #############################################################################################
        ### get unedited 3d coords

        unedited_pixels_idx = np.argwhere(m.cpu().numpy()==False).squeeze(-1)
        unedited_pixels_idx = torch.tensor(unedited_pixels_idx).cuda()

        # print('unedited_pixels_idx', unedited_pixels_idx.shape)
        # edited_pixels_idx = np.argwhere(m.cpu().numpy()==True).squeeze(-1)
        # edited_pixels_idx = torch.tensor(edited_pixels_idx).cuda()


        unedited_pixels_coords = torch.index_select(uv, dim=1, index=unedited_pixels_idx)

        num_samples = unedited_pixels_idx.shape[0]

        depth = torch.ones((batch_size, num_samples)).cuda()
        x_cam = unedited_pixels_coords[:, :, 0].view(batch_size, -1)
        y_cam = unedited_pixels_coords[:, :, 1].view(batch_size, -1)
        z_cam = depth.view(batch_size, -1)

        unedited_pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

        # permute for batch matrix product
        unedited_pixel_points_cam = unedited_pixel_points_cam.permute(0, 2, 1)

        unedited_world_coords = torch.bmm(p, unedited_pixel_points_cam).permute(0, 2, 1)[:, :, :3]
        unedited_ray_dirs = unedited_world_coords - cam_loc[:, None, :]
        unedited_ray_dirs_norm = F.normalize(unedited_ray_dirs, dim=2)

        batch_size, unedited_num_pixels, _ = unedited_ray_dirs_norm.shape
        object_mask = torch.ones((unedited_num_pixels, )).bool().cuda()


        self.implicit_network.eval()
        with torch.no_grad():
            unedited_points, network_object_mask, unedited_dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=unedited_ray_dirs_norm)

        self.implicit_network.train()

        unedited_points = (cam_loc.unsqueeze(1) + unedited_dists.reshape(batch_size, unedited_num_pixels, 1) * unedited_ray_dirs_norm).reshape(-1, 3)

        unedited_differentiable_surface_points = unedited_points[network_object_mask]
        unedited_pixels_idx = unedited_pixels_idx[network_object_mask]

        # print(unedited_pixels_idx.shape)

        # print('network object mask: ', network_object_mask, unedited_differentiable_surface_points.shape)

        # exit()

        unedited_view_dirs = -unedited_ray_dirs_norm.reshape(-1,3)[network_object_mask]



        ###############################################################################################
        world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
        ray_dirs = world_coords - cam_loc[:, None, :]
        ray_dirs_norm = F.normalize(ray_dirs, dim=2)
        
        batch_size, num_pixels, _ = ray_dirs_norm.shape
        object_mask = torch.ones((num_pixels, )).bool().cuda()

        self.implicit_network.eval()
        with torch.no_grad():
            points, _, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs_norm)
        
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs_norm).reshape(-1, 3)


        ###############################################################################################

        differentiable_surface_points = unedited_differentiable_surface_points

        sdf_output = self.implicit_network(points)[:, 0:1] # 通过MLP输出物体几何

        # if edited_pixels_idx.shape[0] == 0:
        #     surface_mask = network_object_mask
        # else:
        #     surface_mask = self.merge_surface_mask(edited_pixels_idx, unedited_pixels_idx, network_object_mask)

        surface_mask = torch.tensor([False, ]*num_samples_0).bool().cuda()
        surface_mask[unedited_pixels_idx] = True

        grad_theta = None

        # print('surface mask: ', surface_mask.shape, surface_mask)


        idr_rgb_values = torch.ones_like(points).float().cuda()
        sg_rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.ones_like(points).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        sg_diffuse_albedo_values = torch.ones_like(points).float().cuda() # diffuse_albedo初始化为1
        sg_specular_rgb_values = torch.zeros_like(points).float().cuda() # specular_rgb初始化为0

        if differentiable_surface_points.shape[0] > 0:

            # print(differentiable_surface_points.shape, unedited_view_dirs.shape)

            ret = self.get_rbg_value_single_mlp(differentiable_surface_points, unedited_view_dirs, flag='unfinetuned', diffuse_rgb=None)

            idr_rgb_values[surface_mask] = ret['idr_rgb']
            sg_rgb_values[surface_mask] = ret['sg_rgb']
            normal_values[surface_mask] = ret['normals']

            sg_diffuse_rgb_values[surface_mask] = ret['sg_diffuse_rgb']
            sg_diffuse_albedo_values[surface_mask] = ret['sg_diffuse_albedo']
            sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb']

        output = {
            'points': points,
            'idr_rgb_values': idr_rgb_values,
            'sg_rgb_values': sg_rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            'network_object_mask': surface_mask,
            'object_mask': surface_mask,
            'grad_theta': grad_theta,
            'sg_diffuse_rgb_values': sg_diffuse_rgb_values,
            'sg_diffuse_albedo_values': sg_diffuse_albedo_values,
            'sg_specular_rgb_values': sg_specular_rgb_values,
            'differentiable_surface_points': differentiable_surface_points,
        }

        return output


    # def merge_surface_mask(self, edited_pixels_idx, unedited_pixels_idx, surface_mask):

    #     length = edited_pixels_idx.shape[0] + unedited_pixels_idx.shape[0]
    #     surface_mask_edited = torch.tensor(False*edited_pixels_idx.shape[0]).cuda()

    #     ### re-sort, since we only take the mask part
    #     tmp = torch.cat((edited_pixels_idx, unedited_pixels_idx), dim=0)
    #     _, idx = tmp.sort(dim=0)
    #     _, rank = idx.sort(dim=0)

    #     edited_pixels_idx = rank[:edited_pixels_idx.shape[0]]
    #     unedited_pixels_idx = rank[edited_pixels_idx.shape[0]:]

    #     print('length: ', length)
    #     print(edited_pixels_idx.shape)
    #     print(unedited_pixels_idx.shape)

    #     mask = torch.zeros((length)).cuda()
    #     mask[edited_pixels_idx] = surface_mask_edited
    #     mask[unedited_pixels_idx] = surface_mask

    #     return mask
