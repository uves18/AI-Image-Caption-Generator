# AI Image Caption Generator

A clean Streamlit app that generates expressive captions for uploaded images using Salesforce's BLIP image-captioning model.

## Overview

This project helps you explore image captioning in multiple tones and formats.
Upload one or more images, choose caption styles, and export the results to CSV.
It runs locally using Hugging Face Transformers and caches the loaded model for faster repeat use.

## Features

- Upload single or multiple images with a friendly Streamlit UI
- Generate captions in multiple styles:
  - Descriptive
  - Detailed
  - Concise
  - Creative / Artistic
  - Social Media
  - SEO / Alt-text
- Display captions next to each image with built-in copy support
- Batch processing with progress feedback
- Download captions as a CSV file

## Requirements

- Python 3.10+ recommended
- Streamlit
- Transformers
- Torch
- Pillow

Install the required dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

The first run may download the BLIP model weights from Hugging Face (about 1 GB for the base model). After that, the model is cached by Streamlit and loads more quickly.

## How it works

The core caption logic lives in `caption_engine.py`.
BLIP supports conditional captioning, so this app uses a small prompt prefix for some styles and light post-processing for others.

Example style behavior:

- `Descriptive (default)`: unconditional BLIP caption
- `Detailed`: caption seeded with `a detailed photography of`
- `Concise`: caption seeded with `a photo of`
- `Creative / Artistic`: caption seeded with `an artistic depiction of`, then styled
- `Social Media`: base caption enhanced with emojis and hashtags
- `SEO / Alt-text`: caption formatted for alt-text use

## Project structure

- `app.py` — Streamlit UI, image upload, style selection, batch processing, CSV export
- `caption_engine.py` — BLIP model loader and caption style generation logic
- `requirements.txt` — Python dependencies
- `README.md` — project documentation

## Notes

- The app automatically chooses GPU if available via PyTorch.
- Streamlit caching keeps the BLIP model loaded between runs.
- Uploaded images are converted to RGB before captioning.

## Future improvements

- Add GPU / device status information in the sidebar
- Use an LLM rewrite step for richer style transformations
- Cache captions by image hash to avoid repeated generation for the same file
- Add unit tests for caption prompt handling and post-processing
- Add a Dockerfile for reproducible deployment

## Demo link
https://drive.google.com/file/d/1MBbWfIIZBZ5-KE2KOQ-Ex3gGTl4CDtVp/view?usp=sharing

## License

This repository is available for personal use and experiment. Feel free to adapt it for your own projects.
