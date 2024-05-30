import sys
sys.path.append('../code')
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import argparse
from pyhocon import ConfigFactory
import numpy as np
import GPUtil
import utils.general as utils
import utils.plots as plt
import imageio
import time
import json
from tqdm import trange

import torch
from torch import nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter

from model.sg_envmap_material import Diffuse_albedo_layer
from model.implicit_differentiable_renderer_v3_cdist import ImplicitNetwork
from skimage.morphology import dilation, erosion, disk, closing, opening

from deepsdf_dataset import SDFSamples


def finetune(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    # expname = conf.get_string('train.expname') + '-' + kwargs['expname']
    expname = kwargs['expname']

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']


    n_epochs = kwargs["n_epochs"]
    lr = kwargs["lr"]
    flag = kwargs["flag"]
    task = kwargs["task"]

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    # evaldir = os.path.join('../cvpr23/exps', conf.get_string('train.expname')+'-'+task, expname, os.path.basename(kwargs['data_split_dir']))

    model_save_dir = os.path.join('../iccv23/exps', task, expname, flag)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    ckpt_save_dir = os.path.join(model_save_dir, 'model_params')
    if not os.path.isdir(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    #########################################################################################################
    ### load model
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    model_dict = model.state_dict()


    feature_vector_size = conf.get_config('model').get_int('feature_vector_size')
    implicit_network = ImplicitNetwork(feature_vector_size, **conf.get_config('model').get_config('implicit_network'))

    diffuse_albedo_layer = Diffuse_albedo_layer(**conf.get_config('model').get_config('diffuse_albedo_layers'))

    if torch.cuda.is_available():
        model.cuda()
        diffuse_albedo_layer.cuda()
        implicit_network.cuda()
    
    
    # old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    # ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    # saved_model_state = torch.load(ckpt_path)
    ckpt_path='/cluster/home/yueshi/code/PointSDF-main/exps/toaster/2024_0527/checkpoints/ModelParameters/latest.pth'
    saved_model_state = torch.load(ckpt_path)
    # only saved_model_state is used afterwards
    new_rendering_network_params = {}
    for key, val in saved_model_state["model_state_dict"].items():
        if key.startswith('envmap_material_network.diffuse'):
            tmp = key.split('.')
            tmp[-2] = 'layer_%s.0' % (int(tmp[-2]) // 2)
            key = ".".join(tmp[1:])
            model_dict[key] = val
        else:
            model_dict[key] = val
            new_rendering_network_params.update({key : val})
    # model.load_state_dict(model_dict)
    print('Loaded checkpoint: ', ckpt_path)

    model_dict = saved_model_state["model_state_dict"]
    # operate on diffuse_albedo_layer
    diffuse_albedo_layer_dict = diffuse_albedo_layer.state_dict()
    for key in diffuse_albedo_layer_dict.keys():
        diffuse_albedo_layer_dict[key] = model_dict['envmap_material_network.'+key]
    diffuse_albedo_layer.load_state_dict(diffuse_albedo_layer_dict, strict=True)
    print(diffuse_albedo_layer_dict.keys())

    # operate on implicit_network
    implicit_network_dict = implicit_network.state_dict()
    for key in implicit_network_dict.keys():
        implicit_network_dict[key] = model_dict['implicit_network.'+key]
    implicit_network.load_state_dict(implicit_network_dict)
    print(implicit_network_dict.keys())
    
    ## export mesh
    with torch.no_grad():
        mesh = plt.get_surface_high_res_mesh(
            sdf=lambda x: implicit_network(x)[:, 0],
            resolution=100
        )

        # Taking the biggest connected component
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_clean = components[areas.argmax()]
        # print(mesh_clean)
     
        # mesh_clean.export('./geometry_editing/output/physg_bear/vis-sdf-origin.obj', 'obj')
        mesh_clean.export('./geometry_editing/vis-sdf-origin.obj', 'obj')
        # it is a ball at begining
    # exit()

    # ### backup unedited network parameters
    # torch.save({"model_state_dict": diffuse_albedo_layer_dict}, os.path.join(ckpt_save_dir, "unedited_diffuse_albedo_layer.pth"))
    # torch.save({"model_state_dict": new_rendering_network_params}, os.path.join(ckpt_save_dir, "envmap_material_network.pth"))


    #######################################################################################################
    dataset = SDFSamples(data_source="load dataset/cluster/home/yueshi/code/PointSDF-main/example_data/shiny/toaster/toaster.npz", subsample=10000)
    # dataset = SDFSamples(data_source="/cluster/home/yueshi/code/PointSDF-main/example_data/bear/bear.npz", subsample=10000)
    # dataset = SDFSamples(data_source="../example_data/nerf_chair/nerf_chair_mesh_fix_arap.npz", subsample=10000)
    # dataset = SDFSamples(data_source='/data2/code_backup/PhySG/code/geometry_editing/data/physg_bear/deformed_mesh_color.npz', subsample=10000)
    # dataset = SDFSamples(data_source='/data2/code_backup/PhySG/code/geometry_editing/data/nerf_chair_remove_leg.npz', subsample=None)

    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=1,
                                                  shuffle=True
                                                  )
    print('Loaded dataset: ', len(dataset))

    
    model.eval()
    
    # model.freeze_idr()
    # # model.envmap_material_network.freeze_all_except_diffuse()
    # model.envmap_material_network.freeze_all()

    # diffuse_albedo_layer_optimizer = torch.optim.Adam(diffuse_albedo_layer.parameters(),
    #                                 lr=lr)
    # diffuse_albedo_layer_scheduler = torch.optim.lr_scheduler.MultiStepLR(diffuse_albedo_layer_optimizer,
    #                                                     # milestones=[0.2*n_epochs, 0.5*n_epochs], #, 0.75*n_epochs],
    #                                                     milestones=[0.8*n_epochs],
    #                                                     gamma=0.2)
    
    implicit_network_optimizer = torch.optim.Adam(implicit_network.parameters(),
                                    lr=lr)
    implicit_network_scheduler = torch.optim.lr_scheduler.MultiStepLR(implicit_network_optimizer,
                                                        milestones=[0.2*n_epochs, 0.5*n_epochs], #, 0.75*n_epochs],
                                                        # milestones=[0.8*n_epochs],
                                                        gamma=0.2)
    
    # loss_l1 = torch.nn.L1Loss(reduction="mean")
    ## clamp l1 loss
    loss_l1 = lambda x, y: torch.mean(torch.clamp(torch.abs(x-y), min=0.0, max=0.1))
    
    import time
    start_time = time.time()
    n_epochs = 70000
    for epoch in range(n_epochs):
        print('Epoch: ', epoch)
        implicit_network.train()

        for data_index, (input, _) in enumerate(train_dataloader):

            pnts = input[0, :, :3].cuda()
            sdf_gt = input[0, :, 3:4].cuda()

            # print('pnts: ', pnts.shape)

            # exit()
            implicit_network_optimizer.zero_grad()

            output = implicit_network(pnts)#[:, 0:1]
            # [:, 0:1]
            # print(output.shape, sdf_gt.shape)
            loss = loss_l1(output, sdf_gt)

            if epoch % 10 == 0:
                print('loss: ', loss.item())
            # print('loss: ', loss.item())
            loss.backward()
            implicit_network_optimizer.step()
            implicit_network_scheduler.step()

    with torch.no_grad():
        mesh = plt.get_surface_high_res_mesh(
            sdf=lambda x: implicit_network(x)[:, 0],
            resolution=100
        )

        # Taking the biggest connected component
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=float)
        mesh_clean = components[areas.argmax()]

        # print(mesh_clean)

     
        mesh_clean.export('./output/toaster/toaster_refined.obj', 'obj')


    ckpt_save_dir = './output/toaster'
    torch.save({"model_state_dict": implicit_network.state_dict()}, os.path.join(ckpt_save_dir, "implicit_network_finetuned.pth"))
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码执行时间为：", execution_time, "秒")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/default.conf')
    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--gamma', type=float, default=1., help='gamma correction coefficient')

    parser.add_argument('--save_exr', default=False, action="store_true", help='')

    parser.add_argument('--light_sg', type=str, default='', help='')
    parser.add_argument('--geometry', type=str, default='', help='')
    parser.add_argument('--diffuse_albedo', type=str, default='', help='')
    parser.add_argument('--view_name', type=str, default='', help='')

    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')

    parser.add_argument('--write_idr', default=False, action="store_true", help='')

    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')

    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    parser.add_argument('--origin_image', type=str, default='', help='')
    parser.add_argument('--edited_image', type=str, default='', help='')
    parser.add_argument('--mask_image', type=str, default='', help='')

    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--flag', type=str, default='finetune_diffuse_albedo')
    parser.add_argument('--task', type=str, default='color_editing')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    finetune(conf=opt.conf,
             write_idr=opt.write_idr,
             gamma=opt.gamma,
             data_split_dir=opt.data_split_dir,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             resolution=opt.resolution,
             save_exr=opt.save_exr,
             light_sg=opt.light_sg,
             geometry=opt.geometry,
             view_name=opt.view_name,
             diffuse_albedo=opt.diffuse_albedo,
             origin_image=opt.origin_image,
             edited_image=opt.edited_image,
             mask_image=opt.mask_image,
             n_epochs=opt.n_epochs,
             lr=opt.lr,
             flag=opt.flag,
             task=opt.task
             )
