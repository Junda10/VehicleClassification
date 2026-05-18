# GitHub Publish Guide

This guide explains how to publish the cleaned repository and the preserved dataset branch to GitHub.

## Current Local Branch Strategy

| Branch | Purpose | Dataset tracking |
| --- | --- | --- |
| `main` | Clean production/portfolio branch | `vehicleClass/` is not tracked |
| `with-dataset` | Original project state with full dataset | `vehicleClass/` is tracked |

## Recommended Push Order

From the repository root:

```bash
cd /Users/terry/Private/VehicleClassification
```

1. Push the preserved dataset branch:

```bash
git push origin with-dataset
```

2. Push the cleaned `main` branch:

```bash
git push origin main
```

3. Confirm branch status:

```bash
git branch -vv
git status --short --branch
```

## Important Notes

- `main` is currently the branch intended for public GitHub presentation.
- `with-dataset` is only for preserving the original dataset-tracked project state.
- The dataset is removed from the latest `main` tree, but old Git history may still contain the dataset because this cleanup does not rewrite history.
- If the goal is the smallest possible public repository, create a new clean GitHub repo from the cleaned `main` working tree or rewrite history before publishing.

## After Pushing

On GitHub:

1. Open the repository page.
2. Confirm the default branch is `main`.
3. Confirm the README renders correctly.
4. Confirm `vehicleClass/` does not appear on the latest `main` branch.
5. Confirm the `with-dataset` branch exists if you want the dataset-preserved branch available remotely.

## Optional: Streamlit Deployment

For Streamlit Community Cloud:

- Repository: `Junda10/VehicleClassification`
- Branch: `main`
- Main file path: `Project.py`

Before deploying, confirm that the two checkpoint files remain available in the `main` branch:

```bash
git ls-files '*.pth'
```

Expected:

```text
best_model.pth
best_model_unfreeze.pth
```
