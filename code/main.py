import argparse, numpy as np
from pathlib import Path
from utils import imread_gray, imread_rgb, imsave
from edges import finite_differences, edges_threshold, dog_edges
from sharpen import unsharp_mask
from hybrid import hybrid
from blend import multires_blend

def save_grid(prefix, **imgs):
    root = Path("outputs")/prefix
    root.mkdir(parents=True, exist_ok=True)
    for name, im in imgs.items():
        imsave(root/f"{name}.jpg", im)
    print(f"[{prefix}] saved -> {root}")

def run_edges(in_path):
    I = imread_gray(in_path)
    Ix, Iy, mag = finite_differences(I)
    edge = edges_threshold(mag)
    save_grid("edges", gray=I, dx=(Ix*0.5+0.5), dy=(Iy*0.5+0.5), grad=mag/mag.max(), edges=edge)

def run_dog(in_path, sigma):
    I = imread_gray(in_path)
    Ix, Iy, mag = dog_edges(I, ksize=int(6*sigma+1)|1, sigma=sigma)
    edge = edges_threshold(mag)
    save_grid("dog", gray=I, dx=(Ix*0.5+0.5), dy=(Iy*0.5+0.5), grad=mag/mag.max(), edges=edge)

def run_sharpen(in_path, sigma, alpha):
    I = imread_gray(in_path)
    S, low, high = unsharp_mask(I, sigma=sigma, alpha=alpha)
    save_grid("sharpen", input=I, low=low, high=(high*0.5+0.5), sharp=S)

def run_hybrid(low_img, high_img, s_low, s_high):
    A = imread_gray(low_img); B = imread_gray(high_img)
    H, L, HP = hybrid(A, B, s_low, s_high)
    save_grid("hybrid", low=A, high=B, lowpass=L, highpass=(HP*0.5+0.5), hybrid=H)

def run_blend(A_path, B_path, mask_path, levels, sigma):
    A = imread_rgb(A_path); B = imread_rgb(B_path); M = imread_gray(mask_path)
    out = multires_blend(A, B, M, levels=levels, sigma=sigma)
    save_grid("blend", A=A, B=B, mask=np.stack([M]*3,-1), blend=out)

def main():
    ap = argparse.ArgumentParser(description="CS180 Project 2 baseline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("edges");   p.add_argument("--input", required=True)
    p = sub.add_parser("dog");     p.add_argument("--input", required=True); p.add_argument("--sigma", type=float, default=1.0)
    p = sub.add_parser("sharpen"); p.add_argument("--input", required=True); p.add_argument("--sigma", type=float, default=1.5); p.add_argument("--alpha", type=float, default=1.0)
    p = sub.add_parser("hybrid");  p.add_argument("--low", required=True); p.add_argument("--high", required=True); p.add_argument("--sigma_low", type=float, default=4.0); p.add_argument("--sigma_high", type=float, default=4.0)
    p = sub.add_parser("blend");   p.add_argument("--A", required=True); p.add_argument("--B", required=True); p.add_argument("--mask", required=True); p.add_argument("--levels", type=int, default=5); p.add_argument("--sigma", type=float, default=2.0)

    args = ap.parse_args()
    if args.cmd=="edges":   run_edges(args.input)
    if args.cmd=="dog":     run_dog(args.input, args.sigma)
    if args.cmd=="sharpen": run_sharpen(args.input, args.sigma, args.alpha)
    if args.cmd=="hybrid":  run_hybrid(args.low, args.high, args.sigma_low, args.sigma_high)
    if args.cmd=="blend":   run_blend(args.A, args.B, args.mask, args.levels, args.sigma)

if __name__ == "__main__":
    main()
