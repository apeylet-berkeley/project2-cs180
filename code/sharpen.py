import numpy as np
from filters import conv2d, gaussian_kernel

def unsharp_mask(img, sigma=1.5, alpha=1.0):
    ksize = int(6*sigma+1) | 1
    g = gaussian_kernel(ksize, sigma)
    low = conv2d(img, g)
    high = img - low
    sharp = np.clip(img + alpha*high, 0, 1)
    return sharp, low, high
