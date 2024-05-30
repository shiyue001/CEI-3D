import os
from glob import glob
import torch

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    # kls--'model.implicit_differentiable_renderer.IDRNetwork'
    # parts--['model', 'implicit_differentiable_renderer', 'IDRNetwork']
    # module--'model.implicit_differentiable_renderer'
    m = __import__(module)
    # m--<class 'model.implicit_differentiable_renderer.IDRNetwork'>
    # parts[1:]--['implicit_differentiable_renderer', 'IDRNetwork']
    for comp in parts[1:]:
        m = getattr(m, comp)
    #m--<class 'model.implicit_differentiable_renderer.IDRNetwork'>
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG', '*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)

        if model_input['diffuse_rgb'] is not None:
            data['diffuse_rgb'] = torch.index_select(model_input['diffuse_rgb'], 0, indx)
        else:
            data['diffuse_rgb'] = None
            
        split.append(data)
    return split


def split_input_idr(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
            
        split.append(data)
    return split


def split_points(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 500
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['differentiable_points'] = torch.index_select(model_input['differentiable_points'], 0, indx)
            
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        print(entry, res[0][entry].shape)
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        elif entry == 'differentiable_surface_points':
            model_outputs[entry] = torch.cat([r[entry] for r in res], 0)
            print(model_outputs[entry].shape)

        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

            # print(model_outputs[entry].shape)

    return model_outputs


def merge_points(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        # print(entry, res[0][entry].shape)
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        elif entry == 'differentiable_surface_points':
            model_outputs[entry] = torch.cat([r[entry] for r in res], 0)
            print(model_outputs[entry].shape)

        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

            # print(model_outputs[entry].shape)

    return model_outputs