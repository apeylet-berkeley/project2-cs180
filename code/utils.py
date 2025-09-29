from pathlib import Path
import numpy as np
import imageio.v3 as iio

def imread_gray(path):
    img = iio.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = 0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]
    if img.max() > 1.0:
        img /= 255.0
    return img

def imread_rgb(path):
    img = iio.imread(path).astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.max() > 1.0:
        img /= 255.0
    return img

def imsave(path, img):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(img, 0, 1)
    out = (out*255.0 + 0.5).astype(np.uint8)
    iio.imwrite(path.as_posix(), out)

def pad_same(img, kshape):
    kh, kw = kshape
    ph, pw = kh//2, kw//2
    return np.pad(img, ((ph,ph),(pw,pw)), mode='edge')
