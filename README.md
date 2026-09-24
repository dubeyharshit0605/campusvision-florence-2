---
title: CampusVision Florence-2
emoji: 🏫
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: 3.11.9
---

# CampusVision

CampusVision is an applied implementation of Microsoft's Florence-2 vision model for
campus-scene understanding. One local Gradio application runs captioning, object detection,
OCR, phrase grounding, and segmentation through the pretrained
`microsoft/Florence-2-base-ft` checkpoint.

This project integrates and evaluates a pretrained model. It does not reproduce the FLD-5B
training pipeline or claim Florence-2 as an original model.

## Features

| Interface option | Florence prompt | Text required | Result |
|---|---|---:|---|
| Caption | `<CAPTION>` | No | Short caption |
| Detailed Caption | `<DETAILED_CAPTION>` | No | Detailed caption |
| More Detailed Caption | `<MORE_DETAILED_CAPTION>` | No | Paragraph caption |
| Object Detection | `<OD>` | No | Labels and boxes |
| OCR | `<OCR>` | No | Recognized text |
| OCR with Regions | `<OCR_WITH_REGION>` | No | Text and quadrilaterals |
| Dense Region Captioning | `<DENSE_REGION_CAPTION>` | No | Region descriptions and boxes |
| Phrase Grounding | `<CAPTION_TO_PHRASE_GROUNDING>` | Yes | Phrase locations |
| Referring Expression Segmentation | `<REFERRING_EXPRESSION_SEGMENTATION>` | Yes | Requested-object polygons |
| Open Vocabulary Detection | `<OPEN_VOCABULARY_DETECTION>` | Yes | Requested categories and regions |

The interface accepts upload, webcam, and clipboard images. Spatial results appear on an
annotated image, while generated text and parsed JSON remain available separately. Each run
saves downloadable PNG and JSON artifacts and adds a row to `outputs/experiments.csv`.

## Local setup

Python 3.11 is the validated runtime. Python 3.14 is outside the supported range because the
tested PyTorch and Florence-2 dependency set does not target it.

### Windows PowerShell, CPU

```powershell
py install 3.11 -y
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\campusvision.exe
```

For an NVIDIA machine, install the PyTorch build shown by the official PyTorch selector for
the installed CUDA version before installing this project. CampusVision selects CUDA first,
then Apple MPS, then CPU. The smoke command and Python API also allow an explicit device.

### macOS, Apple Silicon or CPU

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
campusvision
```

The first real run downloads the pinned checkpoint files into the Hugging Face cache. Later
runs reuse that cache. Model loading time is reported separately from inference time.
`requirements-lock.txt` records the validated Windows CPU environment; its `+cpu` PyTorch
build is an audit snapshot, while `pyproject.toml` is the cross-platform installation source.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Unit tests use a controlled backend and never present its responses as model predictions.
Run a real checkpoint smoke test with the included synthetic fixture:

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_test `
  --image tests\fixtures\campus_notice.png `
  --task Caption `
  --device cpu `
  --max-new-tokens 64 `
  --num-beams 1
```

Phrase tasks add `--text`, for example `--task "Phrase Grounding" --text "the noticeboard"`.

## Hugging Face Spaces deployment

Create a Gradio Space, then push this repository to the Space repository. The README metadata,
root `app.py`, and `requirements.txt` provide the required entry point and dependency install.
The default Space runs on CPU. Select suitable GPU hardware in Space settings when faster
inference is needed. Keep the Space private when uploaded campus images must not be public.

The model code is loaded from the immutable checkpoint revision
`f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`. Review that revision before changing it because
the checkpoint uses trusted remote Python code.

## Output layout

```text
outputs/
  experiments.csv
  <run-id>/
    result.json
    annotated.png
```

The CSV records UTC time, checkpoint and revision, dependency versions, task, phrase,
original and processed image dimensions, generation settings, actual device and dtype,
model-load time, synchronized inference time, total time, process RSS before and after,
accelerator memory when available, status, error, and artifact paths.

Process RSS is the memory used by the whole Python process, not isolated model memory or a
guaranteed peak. CUDA inference is synchronized around timing. A cold run includes model
initialization; compare machines using warm runs.

## Reproducible evaluation

Use the same checkpoint revision, locked dependencies, input file, processed dimensions,
task, phrase, token limit, and beam count on every machine. Run one cold trial, then at least
five warm trials per task and report median and spread. Compare Mac MPS and Windows CPU only
from measurements collected on those machines.

Accuracy requires a labelled dataset. Store expected captions, text, boxes, or masks outside
the smoke fixture and choose task-appropriate metrics before making accuracy claims. The
included image only proves that the application and model can complete an end-to-end run.

## Limitations

- Captions can hallucinate details.
- Small, blurry, rotated, or handwritten text can reduce OCR quality.
- Detection and grounding can miss requested objects.
- Generated polygons are not guaranteed to be pixel-accurate masks.
- CPU inference is substantially slower than accelerator inference.
- Results vary with image quality, prompt wording, token limit, and beam count.
