import PIL
import numpy as np
import config as cg

def load_data(path='', shape=None, mode='eval'):

    short_size = 224.0
    crop_size = shape
    img = PIL.Image.open(path)
    im_shape = np.array(img.size)    # in the format of (width, height, *)
    img = img.convert('RGB')

    ratio = float(short_size) / np.min(im_shape)
    img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),   # width
                           int(np.ceil(im_shape[1] * ratio))),  # height
                     resample=PIL.Image.BILINEAR)

    x = np.array(img)  # image has been transposed into (height, width)
    newshape = x.shape[:2]
    if mode == 'eval':    # center crop
        h_start = (newshape[0] - crop_size[0])//2
        w_start = (newshape[1] - crop_size[1])//2
    else:
        raise IOError('==> unknown mode.')
    x = x[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    x = x[:, :, ::-1] - cg.mean
    return x

