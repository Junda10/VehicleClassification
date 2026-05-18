# Repository Cleanup Plan

This document tracks cleanup decisions needed before presenting the project publicly on GitHub or deploying the app.

## Current Size Risks

The repository currently includes large artifacts:

| Artifact | Approx. size | Notes |
| --- | ---: | --- |
| `vehicleClass/` | ~937 MB | Full train/validation/test image dataset |
| `best_model.pth` | ~90 MB | ResNet50 baseline checkpoint |
| `best_model_unfreeze.pth` | ~90 MB | Fine-tuned ResNet50 checkpoint |
| Notebooks | ~8 MB each | Include outputs/plots; acceptable but can be cleaned |

These files make the repository slow to clone and harder to deploy on app-hosting platforms.

## Recommended Public GitHub Strategy

For the cleanest portfolio repository:

1. Keep code, docs, notebooks, and a few small demo assets in Git.
2. Move full dataset files out of the main repository.
3. Move model checkpoints to a release/artifact host.
4. Add download instructions or startup download logic for model weights.

## Artifact Hosting Options

| Option | Best for | Pros | Cons |
| --- | --- | --- | --- |
| GitHub Releases | Model checkpoints | Easy to link from README; keeps main repo lighter | App must download files or user must download manually |
| Git LFS | Model checkpoints, small curated datasets | Git-friendly workflow for large files | Requires Git LFS setup; storage/bandwidth limits may apply |
| Hugging Face Hub | ML checkpoints and datasets | Designed for ML artifacts; pairs well with Spaces | Requires account/repo setup |
| External cloud storage | Full dataset | Flexible | Less polished for portfolio unless documented clearly |

## Selected Decision

Owner-selected strategy for this cleanup pass:

- **Code/docs/notebooks:** keep in GitHub repo.
- **Full dataset (`vehicleClass/`):** remove from Git tracking and keep ignored locally.
- **Model checkpoints (`*.pth`):** keep in the repository for now so the Streamlit app can run without extra download logic.
- **Demo images/screenshots:** keep a small `assets/` folder in Git when demo assets are created.

Implementation note: `git rm -r --cached vehicleClass` removes the dataset from Git tracking without deleting the local files from disk. `.gitignore` prevents the dataset from being re-added accidentally.

Important Git history note: this removes the dataset from the latest commit/tree, but it does not rewrite old Git history. If the goal is the smallest possible public repository, create a fresh clean GitHub repository from the cleaned working tree or rewrite history with a tool such as `git filter-repo` before publishing.

## Possible Next Implementation Steps

### Option A — Keep current repo as-is for now

Use this if the immediate goal is only improving README/docs before pushing.

- Fastest path.
- Keeps app working locally without download logic.
- Repo remains large and less production-like.

### Option B — Move checkpoints to GitHub Releases

- Upload `best_model.pth` and `best_model_unfreeze.pth` to a GitHub Release.
- Add URLs to app config.
- Update `Project.py` to download missing checkpoints into a local `models/` folder.
- Stop tracking `.pth` files in the main branch.

### Option C — Move model + demo to Hugging Face Spaces

- Create a Hugging Face Space for the Streamlit app.
- Store model artifacts in the Space or Hugging Face Hub.
- Link the live demo from the GitHub README.

## Completed / Pending Decisions

- [x] Full dataset should not remain tracked in the public GitHub repo.
- [x] Model checkpoints should remain in Git for now.
- [x] Public repo should prioritize code, documentation, notebooks, and future small demo assets.
- [ ] Optional future step: move model checkpoints to GitHub Releases or Hugging Face Hub if repository size or deployment limits become a problem.
