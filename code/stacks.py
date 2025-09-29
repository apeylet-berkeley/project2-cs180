from filters import conv2d, gaussian_kernel

def _blur_any(img, sigma):
    ksize = int(6*sigma+1) | 1
    g = gaussian_kernel(ksize, sigma)
    return conv2d(img, g)

def gaussian_stack(img, levels=5, sigma=2.0):
    stack = [img]
    cur = img
    for _ in range(1, levels):
        cur = _blur_any(cur, sigma)
        stack.append(cur)
    return stack

def laplacian_stack(img, levels=5, sigma=2.0):
    G = gaussian_stack(img, levels, sigma)
    L = []
    for i in range(len(G)-1):
        L.append(G[i] - G[i+1])
    L.append(G[-1])
    return L
