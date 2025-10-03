# CS180 Project 2 — Filtering & Hybrid Images (Website)

This repo contains a static site to showcase results for:
- 1.1 Convolutions from Scratch
- 1.2 Finite Difference Operator
- 1.3 Derivative of Gaussian (DoG)
- 2.1 Unsharp Mask (Sharpening)
- 2.2 Hybrid Images (with a short reflection)
- 2.3 Gaussian & Laplacian Stacks
- 2.4 Multiresolution Blending (Oraple)

## How to use

1. Export images from your notebook and drop them in `assets/img/`.
2. Ensure filenames in `index.html` match your exported images.
3. Open `index.html` locally or deploy with GitHub Pages.

## GitHub Pages

- Create a public repo (e.g., `cs180-proj2-site`) and push this folder.
- In GitHub → Settings → Pages:
  - Source: `Deploy from a branch`
  - Branch: `main` (or `master`), folder `/root`
- Your site will be available at: `https://<your-username>.github.io/cs180-proj2-site/`

## Notes
- Keep all assets inside `assets/img/`.
- You can add more figures by duplicating `<figure>...</figure>` blocks.
- The CSS aims for a clean, minimal aesthetic similar to common CS project pages.

## License
Choose any license you prefer (MIT recommended).
