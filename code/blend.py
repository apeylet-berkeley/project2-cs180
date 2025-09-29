import numpy as np
from stacks import gaussian_stack, laplacian_stack

def _center_crop_hw(img, h, w):
    H, W = img.shape[:2]
    y0 = (H - h)//2
    x0 = (W - w)//2
    if img.ndim == 2:
        return img[y0:y0+h, x0:x0+w]
    else:
        return img[y0:y0+h, x0:x0+w, :]

def _match_three(A, B, M):
    h = min(A.shape[0], B.shape[0], M.shape[0])
    w = min(A.shape[1], B.shape[1], M.shape[1])
    return _center_crop_hw(A, h, w), _center_crop_hw(B, h, w), _center_crop_hw(M, h, w)

def multires_blend(A, B, mask, levels=5, sigma=2.0):
    # Ensure shapes align (center-crop to common HxW)
    A, B, mask = _match_three(A, B, mask)

    LA = laplacian_stack(A, levels, sigma)
    LB = laplacian_stack(B, levels, sigma)
    GM = gaussian_stack(mask, levels, sigma)

    LS = []
    for l in range(levels):
        M = GM[l]
        if M.ndim == 2 and LA[l].ndim == 3:
            M = M[..., None]
        LS.append(M*LA[l] + (1.0 - M)*LB[l])

    out = np.zeros_like(LS[0])
    for l in reversed(range(levels)):
        out = out + LS[l]
    return np.clip(out, 0, 1)
