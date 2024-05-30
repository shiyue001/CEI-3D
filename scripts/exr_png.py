import os
import imageio
import numpy as np
from PIL import Image
# to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)  #use for 0~1
path = '/cluster/home/yueshi/code/PointSDF-main/example_data/kitty/train/image'
path_png = '/cluster/home/yueshi/code/PointSDF-main/example_data/kitty/train/1/'
for exr_name in os.listdir(path):
    png_name, _ = exr_name.split('.')
    exr_path = os.path.join(path, exr_name)
    exr_file = imageio.imread(exr_path)
    tonemap_img = lambda x: np.power(x, 1./2.2)
    clip_img = lambda x: np.clip(x,0.,1.)
    image = clip_img(tonemap_img(exr_file))
    # image = image[:,:,:3].astype(np.uint8)*255
    img = Image.fromarray((image*255).astype(np.uint8)).convert('RGB')
    img.save(path_png+png_name+'.png')
    # exr_file = exr_file[:,:,:3]
    # imageio.imwrite(path_png+png_name+'.png',exr_file)
    
    