# Demo Assets Guide

Use this guide to create screenshots or a short demo GIF for the GitHub README after the Streamlit app runs locally.

## Run the App Locally

```bash
streamlit run Project.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Recommended README Screenshot

Capture one clean screenshot showing:

- App title: `Vehicle Classification App`
- Model dropdown
- Uploaded vehicle image
- Prediction result and confidence score

Suggested save path:

```text
assets/demo-screenshot.png
```

After adding the screenshot, embed it in `README.md`:

```markdown
![Vehicle Classification demo](assets/demo-screenshot.png)
```

## Optional Demo GIF

A short GIF can make the repository more portfolio-friendly.

Recommended flow:

1. Open the app.
2. Select the fine-tuned ResNet50 model.
3. Upload a sample vehicle image.
4. Show the predicted class and confidence score.

Suggested save path:

```text
assets/demo.gif
```

Embed it in `README.md`:

```markdown
![Vehicle Classification app demo](assets/demo.gif)
```

## Sample Images

For public GitHub presentation, use only images that are safe to redistribute. Good options:

- Use a few images from the existing dataset only if the dataset license permits redistribution.
- Use your own photos.
- Use public-domain or permissively licensed images and credit the source.

Recommended folder:

```text
assets/sample-images/
```

Avoid committing the full dataset just for demo images.

## Screenshot Checklist

- [ ] App runs successfully with `streamlit run Project.py`.
- [ ] Screenshot does not show private file paths or personal information.
- [ ] Uploaded image license is safe for public GitHub use.
- [ ] Screenshot/GIF is compressed to a reasonable size before committing.
- [ ] README includes the screenshot/GIF near the top of the file.
