# Deployment Plan

This project currently runs as a local Streamlit app (`Project.py`) with PyTorch model checkpoints. The best deployment path is an ML app host, not GitHub Pages.

## Hosting Options

| Option | Fit | Notes |
| --- | --- | --- |
| GitHub Pages | Static landing page only | GitHub Pages cannot run Python/PyTorch/Streamlit inference. Use it for documentation, screenshots, and links. |
| Streamlit Community Cloud | Recommended for current app | Minimal changes because the project already uses Streamlit. Best first deployment target. |
| Hugging Face Spaces | Strong ML demo option | Good if model files are hosted with the Space or downloaded from Releases/Hugging Face Hub. |
| Render/Fly.io/Railway | Possible but heavier | More control, but requires app/server configuration and usually paid resources for reliable ML inference. |

## Recommended Approach

1. Keep GitHub as the polished portfolio repository.
2. Use the README as the project landing page.
3. Deploy the interactive demo on Streamlit Community Cloud or Hugging Face Spaces.
4. Optionally create a GitHub Pages static page later that links to the live demo.

## Changes Needed Before Hosting

### 1. Remove runtime dependency on the training dataset — done

`Project.py` now uses a fixed `CLASS_NAMES` list, so the app can run without scanning `vehicleClass/train/` at runtime.

### 2. Use deterministic inference transforms — done

The inference pipeline now uses deterministic resize/crop/normalize preprocessing instead of random training augmentations.

### 3. Reduce repository size

The repository contains a full dataset and large model checkpoints. This makes GitHub cloning and cloud deployment slower.

Recommended options:

- Keep only a few sample images in the repo.
- Move full datasets outside GitHub.
- Move model checkpoints to GitHub Releases, Git LFS, or Hugging Face Hub.
- Download checkpoints at app startup if they are not present locally.

### 4. Clean dependencies — first pass done

`requirements.txt` now contains only app runtime dependencies. Notebook/training packages are listed separately in `requirements-training.txt`.

## Streamlit Community Cloud Checklist

- [x] Refactor app so it does not require `vehicleClass/train/` at runtime.
- [ ] Confirm model checkpoint file strategy: bundled file, GitHub Release download, Git LFS, or Hugging Face Hub.
- [ ] Confirm `requirements.txt` installs cleanly on Linux.
- [ ] Push the repository to GitHub.
- [ ] In Streamlit Community Cloud, choose this repo and set the main file path to `Project.py`.
- [ ] Test image upload and prediction from the deployed URL.

## GitHub Pages Checklist

Use GitHub Pages only if creating a static project website:

- [ ] Add a `docs/` or `site/` folder with static HTML/Markdown content.
- [ ] Include screenshots or GIFs of the Streamlit app.
- [ ] Link to the Streamlit/Hugging Face live demo.
- [ ] Enable Pages in repository settings.
