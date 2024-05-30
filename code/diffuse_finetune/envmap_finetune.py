import sys
sys.path.append('../code')

import os
import argparse
from pyhocon import ConfigFactory
import numpy as np
import GPUtil
import utils.general as utils
import utils.plots as plt
import imageio
import time
import json

import torch
from torch import nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter

from model.sg_envmap_material import Diffuse_albedo_layer
from skimage.morphology import dilation, erosion, disk, closing, opening

### split and merge, edited on 2022.11.03

def get_img_loss(loss_type='L1'):
    if loss_type == 'L1':
        # print('Using L1 loss for comparing images!')
        img_loss = nn.L1Loss(reduction='mean')
    elif loss_type == 'L2':
        # print('Using L2 loss for comparing images!')
        img_loss = nn.MSELoss(reduction='mean')

    return img_loss



def get_diffuse_loss(diffuse_rgb_values, rgb_gt, network_object_mask, object_mask, img_loss=get_img_loss(loss_type='L1')):
    mask = network_object_mask & object_mask
    if mask.sum() == 0:
        return torch.tensor(0.0).cuda().float()

    diffuse_rgb_values = diffuse_rgb_values[mask].reshape((-1, 3))
    rgb_gt = rgb_gt.reshape(-1, 3)[mask].reshape((-1, 3))

    sg_rgb_loss = img_loss(diffuse_rgb_values, rgb_gt)

    return sg_rgb_loss


def finetune(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    # expname = conf.get_string('train.expname') + '-' + kwargs['expname']
    expname = kwargs['expname']

    print(os.path.join('../', kwargs['exps_folder_name'], '0_unedited_models', expname))

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], '0_unedited_models', expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], '0_unedited_models', expname))
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
    expdir = os.path.join('../', exps_folder_name, '0_unedited_models', expname)
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

    diffuse_albedo_layer = Diffuse_albedo_layer(multires=10, dims=[512, 512, 512, 512])
    if torch.cuda.is_available():
        model.cuda()
        diffuse_albedo_layer.cuda()


    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    saved_model_state = torch.load(ckpt_path)
    # model.load_state_dict(saved_model_state["model_state_dict"])

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


    model.load_state_dict(model_dict)

    print('Loaded checkpoint: ', ckpt_path)

    model_dict = saved_model_state["model_state_dict"]
    diffuse_albedo_layer_dict = diffuse_albedo_layer.state_dict()
    for key in diffuse_albedo_layer_dict.keys():
        diffuse_albedo_layer_dict[key] = model_dict['envmap_material_network.'+key]

    diffuse_albedo_layer.load_state_dict(diffuse_albedo_layer_dict, strict=True)

    print(diffuse_albedo_layer_dict.keys())

    # exit()

    # ### backup unedited network parameters
    # torch.save({"model_state_dict": diffuse_albedo_layer_dict}, os.path.join(ckpt_save_dir, "unedited_diffuse_albedo_layer.pth"))
    # torch.save({"model_state_dict": new_rendering_network_params}, os.path.join(ckpt_save_dir, "envmap_material_network.pth"))


    #######################################################################################################
    ## load dataset
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                           kwargs['data_split_dir'],
                                                                           train_cameras=False)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=eval_dataset.collate_fn
                                                  )

    total_pixels = eval_dataset.total_pixels

    ########################################################################################################
    ### load origin image and edited image

    edited_image = imageio.imread(kwargs['edited_image']).astype(np.float32)[:, :, :3]
    mask_image = imageio.imread(kwargs['mask_image'])

    ### TODO: get edited 2D points position according to the origin image and the edited one
    ### size: batch_size x n_points x 2
    # diff = np.sum((edited_image-origin_image)**2, axis=-1)
    # edited_area = np.equal(diff, 0.).astype(np.uint8)


    ########################################################################################################
    ### texture editing
    # ones = np.ones_like(diff)
    # zeros = np.zeros_like(diff)

    # # diff_img = np.equal(diff_img, 0).astype(np.uint8)
    # # edited_area = np.where(diff<100, zeros, ones).astype(np.uint8)*255

    # edited_area = np.where(diff==0, zeros, ones).astype(np.uint8)*255
    ########################################################################################################


    ########################################################################################################
    ### color editing
    # ones = np.ones_like(diff)
    # zeros = np.zeros_like(diff)

    # edited_area = np.where(diff<20, zeros, ones).astype(np.uint8)

    # # kernel = disk(4)
    # # edited_area = opening(edited_area, kernel)

    # # kernel = disk(6)
    # # edited_area = closing(edited_area, kernel)

    # edited_area = edited_area * 255

    # ones = np.ones_like(mask_image) * 255.
    # zeros = np.zeros_like(mask_image)

    # edited_area = np.where(np.equal(mask_image, ones), zeros, ones).astype(np.uint8)#[:,:,0]

    edited_area = mask_image

    print('edited_area shape: ', edited_area.shape)

    ########################################################################################################

    imageio.imsave(os.path.join(model_save_dir, 'edited_2d_pixels.png'), edited_area)

    ### convert uv coords to image coords
    edited_area = np.transpose(edited_area)#[:,::-1]

    edited_2d_positions = np.argwhere(edited_area).astype(np.float32)
    edited_2d_positions = torch.from_numpy(edited_2d_positions).unsqueeze(0).cuda()


    if not kwargs['edited_image'].endswith('.exr'):
        edited_image /= 255.
    rgb_images = torch.from_numpy(edited_image).cuda().reshape((-1, 3))

    rgb_gt = rgb_images

    model.eval()
    model.freeze_idr()
    # model.envmap_material_network.freeze_all_except_diffuse()
    model.envmap_material_network.freeze_all()

    diffuse_albedo_layer_optimizer = torch.optim.Adam(diffuse_albedo_layer.parameters(),
                                    lr=lr)
    diffuse_albedo_layer_scheduler = torch.optim.lr_scheduler.MultiStepLR(diffuse_albedo_layer_optimizer,
                                                        milestones=[0.3*n_epochs, 0.5*n_epochs, 0.75*n_epochs],
                                                        # milestones=[0.5*n_epochs, 0.75*n_epochs],
                                                        gamma=0.1)

    # utils.mkdir_ifnotexists(evaldir)
    # print('Output directory is: ', evaldir)

    # with open(os.path.join(evaldir, 'ckpt_path.txt'), 'w') as fp:
    #     fp.write(ckpt_path + '\n')


    if kwargs['light_sg'].endswith('.npy'):
        print('Loading light from: ', kwargs['light_sg'])

        # 环境光贴图
        envmap, _ = os.path.split(kwargs['light_sg'])
        _, envmap, = os.path.split(envmap)

        model.envmap_material_network.load_light(kwargs['light_sg'])


    ret = []

    writer = SummaryWriter(model_save_dir)
    start = time.time()

    
    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):

        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()
        model_input['pose'] = model_input['pose'].cuda()
        model_input['diffuse_rgb'] = None

        print('total_pixels: ', total_pixels)
        split = utils.split_input(model_input, total_pixels)
        res = []

        for s in split:
            out = model.get_surface_points(s)
            res.append({
                'points' : out['points'], 
                'differentiable_surface_points' : out['differentiable_surface_points'],
                'network_object_mask' : out['network_object_mask'],
                'object_mask' : out['object_mask']
            })

        model_outputs = utils.merge_output(res, total_pixels, batch_size=1)

        points = model_outputs['points']
        differentiable_points = model_outputs['differentiable_surface_points']
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']


        edited_3d_points = model.get_edited_3d_points(model_input, edited_2d_positions)

        print(edited_3d_points.shape)

        ### To save the edited 3d locations
        np.save(os.path.join(model_save_dir, 'edited_3d_points'), edited_3d_points.cpu().numpy())

        print(model_save_dir)

        for epoch in range(n_epochs):

            sg_diffuse_albedo_values = torch.ones_like(points).float().cuda()
           
            output = diffuse_albedo_layer(differentiable_points)
            diffuse_albedo = output['sg_diffuse_albedo']

            sg_diffuse_albedo_values[network_object_mask] = diffuse_albedo


            # loss = get_diffuse_loss(diffuse_rgb, rgb_gt, network_object_mask, object_mask, img_loss=get_img_loss(loss_type='L1'))
            loss = get_diffuse_loss(sg_diffuse_albedo_values, rgb_gt, network_object_mask, object_mask, img_loss=get_img_loss(loss_type='L1'))
            

            ### add regularization loss
            # weight_change_loss = 0.
            # for k, v in diffuse_albedo_layer.state_dict().items():
            #     if 'weight' in k:
            #         diff = (model_dict['envmap_material_network.'+k] - v).pow(2).mean()
            #         weight_change_loss += diff
            # loss = loss + 0.5 * weight_change_loss


            diffuse_albedo_layer_optimizer.zero_grad()

            loss.backward()

            print('\n', loss.item())
            ret.append(loss.item())

            diffuse_albedo_layer_optimizer.step()

            diffuse_albedo_layer_scheduler.step()

            writer.add_scalar('loss', loss.item(), epoch)


        model_dict = model.state_dict()
        diffuse_albedo_layer_dict = diffuse_albedo_layer.state_dict()

        for key, val in diffuse_albedo_layer_dict.items():
            tmp = key.split('.')
            tmp[-2] = 'layer_%s.0' % (int(tmp[-2]) // 2)
            tmp[0] = '%s_finetuned' % tmp[0]
            key = ".".join(tmp)
            model_dict[key] = val
                

        torch.save(
                {"epoch": epoch, "model_state_dict": model_dict},
                os.path.join(ckpt_save_dir, "%s_model.pth" % epoch))
        torch.save(
                {"model_state_dict": model_dict},
                os.path.join(ckpt_save_dir, "latest_model.pth"))

    end = time.time()
    print(ret)

    print('time: ', end-start)
    with open(os.path.join(model_save_dir, 'log.txt'), 'a') as f:
        f.write('time: %d\nloss:\n' % (end-start))
        for loss in ret:
            f.write('%s\n' % str(loss))


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
             edited_image=opt.edited_image,
             mask_image=opt.mask_image,
             n_epochs=opt.n_epochs,
             lr=opt.lr,
             flag=opt.flag,
             task=opt.task
             )
