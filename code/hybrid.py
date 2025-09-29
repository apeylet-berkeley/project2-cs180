import numpy as np
from filters import conv2d, gaussian_kernel

def lowpass(img, sigma):
    ksize = int(6*sigma+1) | 1
    g = gaussian_kernel(ksize, sigma)
    return conv2d(img, g)

def highpass(img, sigma):
    return img - lowpass(img, sigma)

def _center_crop_2d(img, h, w):
    H, W = img.shape
    y0 = (H - h)//2
    x0 = (W - w)//2
    return img[y0:y0+h, x0:x0+w]

def hybrid(low_img, high_img, sigma_low=4.0, sigma_high=4.0, mode="crop"):
    """
    Combine low frequencies of low_img with high frequencies of high_img.
    If shapes differ, default to center-cropping to the common size.
    If mode='resize' and Pillow is available, resize high to low.
    """
    L = lowpass(low_img, sigma_low)
    H = highpass(high_img, sigma_high)

    if L.shape != H.shape:
        if mode == "resize":
            try:
                from PIL import Image
                H8 = (H*255.0 + 0.5).astype(np.uint8)
                H = np.array(Image.fromarray(H8).resize((L.shape[1], L.shape[0]), Image.BILINEAR)).astype(np.float32)/255.0
            except Exception:
                # fallback to crop
                mode = "crop"

        if mode == "crop":
            h = min(L.shape[0], H.shape[0])
            w = min(L.shape[1], H.shape[1])
            L = _center_crop_2d(L, h, w)
            H = _center_crop_2d(H, h, w)

    hyb = np.clip(L + H, 0, 1)
    return hyb, L, H
