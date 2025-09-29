import numpy as np
from filters import conv2d, derive_kernels, dog_kernels

def finite_differences(img):
    kx, ky = derive_kernels()
    Ix = conv2d(img, kx)
    Iy = conv2d(img, ky)
    mag = np.sqrt(Ix**2 + Iy**2)
    return Ix, Iy, mag

def edges_threshold(mag, thresh=None):
    if thresh is None:
        thresh = 0.5*mag.mean() + 0.5*np.median(mag)
    return (mag > thresh).astype(np.float32)

def dog_edges(img, ksize=7, sigma=1.0):
    gx, gy = dog_kernels(ksize, sigma)
    Ix = conv2d(img, gx)
    Iy = conv2d(img, gy)
    mag = np.sqrt(Ix**2 + Iy**2)
    return Ix, Iy, mag
