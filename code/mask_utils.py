from pathlib import Path
import numpy as np
import imageio.v3 as iio

def vertical_mask(h, w, feather=40):
    m = np.zeros((h,w), np.float32)
    m[:, :w//2] = 1.0
    if feather>0:
        x = np.linspace(-feather, feather, 2*feather+1)
        ramp = (0.5*(1 - x/feather)).clip(0,1)
        ramp = ramp[feather:]  # right half
        for i in range(min(feather, w-w//2)):
            m[:, w//2+i] = ramp[i]
    return m

def circular_mask(h, w, radius=None, feather=30):
    y,x = np.ogrid[:h,:w]
    cy,cx = h//2, w//2
    if radius is None: radius = min(h,w)//3
    dist = np.sqrt((y-cy)**2 + (x-cx)**2)
    m = (dist <= radius).astype(np.float32)
    if feather>0:
        ring = (dist>radius) & (dist<radius+feather)
        m[ring] = np.maximum(0.0, 1 - (dist[ring]-radius)/feather)
    return m

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["vertical","circle"], required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--feather", type=int, default=40)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    if args.type=="vertical":
        m = vertical_mask(args.height, args.width, args.feather)
    else:
        m = circular_mask(args.height, args.width, radius=None, feather=args.feather)
    out = (np.clip(m,0,1)*255+0.5).astype(np.uint8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, out)
    print("wrote", args.output)
