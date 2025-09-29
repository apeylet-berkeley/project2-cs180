import numpy as np

def pad_same(img, kshape):
    kh, kw = kshape
    ph, pw = kh//2, kw//2
    return np.pad(img, ((ph,ph),(pw,pw)), mode='edge')

def _conv2d_gray(img, kernel):
    img = img.astype(np.float32)
    k = np.flipud(np.fliplr(kernel.astype(np.float32)))
    kh, kw = k.shape
    padded = pad_same(img, (kh, kw))
    H, W = img.shape
    out = np.zeros((H, W), np.float32)
    for y in range(H):
        ys, ye = y, y+kh
        for x in range(W):
            xs, xe = x, x+kw
            patch = padded[ys:ye, xs:xe]
            out[y, x] = np.sum(patch * k)
    return out

def conv2d(img, kernel):
    """2D convolution; if img is RGB (H,W,3), convolve each channel."""
    if img.ndim == 2:
        return _conv2d_gray(img, kernel)
    elif img.ndim == 3:
        chans = [ _conv2d_gray(img[..., c], kernel) for c in range(img.shape[-1]) ]
        return np.dstack(chans)
    else:
        raise ValueError("conv2d expects 2D (H,W) or 3D (H,W,C) image")

def gaussian_kernel(ksize=7, sigma=1.0):
    ax = np.arange(-(ksize//2), ksize//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    g /= np.sum(g)
    return g

def derive_kernels():
    kx = np.array([[1, -1]], dtype=np.float32)
    ky = kx.T
    return kx, ky

def dog_kernels(ksize=7, sigma=1.0):
    g = gaussian_kernel(ksize, sigma)
    kx, ky = derive_kernels()
    gx = conv2d(g, kx)   # G * Dx
    gy = conv2d(g, ky)   # G * Dy
    return gx, gy
