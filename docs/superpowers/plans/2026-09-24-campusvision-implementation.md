# CampusVision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Gradio application that runs `microsoft/Florence-2-base-ft` across the requested campus-scene vision tasks, renders spatial results, and exports reproducible run artifacts and measurements.

**Architecture:** A typed task registry validates user intent, a lazy singleton inference service owns the processor and model, pure renderers draw parsed spatial outputs, and an artifact store persists JSON, PNG, and CSV records. The Gradio layer remains thin so every behavior except browser wiring can be tested without loading the model.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, Gradio, Pillow, psutil, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-24-campusvision-design.md`

## Global Constraints

- Use the pretrained checkpoint `microsoft/Florence-2-base-ft`; do not train Florence-2 or present inference integration as model reproduction.
- Support CUDA, Apple MPS, and CPU with an explicit device override and record the device and dtype actually used.
- Provide caption, detailed caption, more detailed caption, object detection, OCR, OCR with regions, dense region captioning, phrase grounding, referring-expression segmentation, and open-vocabulary detection.
- Require nonempty text for phrase-based tasks and bound image dimensions, phrase length, and generated token count.
- Serialize inference through one cached model instance and keep public Gradio sharing disabled by default.
- Keep coordinates tied to the processed image, while recording both original and processed dimensions.
- Record cold model-load time separately from synchronized warm inference time; describe RSS as process memory rather than model-only memory.
- Save unique PNG and JSON artifacts per run and append one consistent, serialized CSV record for each successful or failed run.
- Never return a mock prediction as a real result; test doubles are confined to automated tests.
- Treat model-generated captions, OCR, detections, and polygons as fallible outputs, not ground truth.

## Review Focus

- EXIF-rotated, palette, RGBA, and grayscale images must normalize to correctly oriented RGB without coordinate drift; Task 2 tests this.
- Empty or whitespace-only text for phrase tasks must fail before model loading, while fixed-prompt tasks must ignore stale UI text; Task 1 tests both paths.
- Nested polygon output, OCR quadrilaterals, missing labels, and coordinates outside the canvas must render safely without mutating the source image; Task 2 tests these shapes.
- Concurrent runs must not interleave model generation or corrupt CSV rows; Tasks 3 and 4 test writer and inference locks.
- Unsupported accelerator overrides and device runtime failures must produce actionable errors and must never be silently logged as successful inference; Tasks 3 and 4 test both conditions.

---

### Task 1: Project foundation and task contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/campusvision/__init__.py`
- Create: `src/campusvision/contracts.py`
- Create: `src/campusvision/tasks.py`
- Create: `tests/test_tasks.py`

**Interfaces:**
- Consumes: No application interfaces.
- Produces: `TaskSpec`, `TaskInputError`, `TASKS`, `get_task(task_name: str) -> TaskSpec`, and `build_prompt(task_name: str, text_input: str | None) -> tuple[TaskSpec, str]`.

- [ ] **Step 1: Create the package configuration and failing task tests**

Create `pyproject.toml` with Python `>=3.11,<3.13`, a `src` package layout, runtime dependencies for `torch`, `transformers`, `gradio`, `pillow`, `psutil`, and `numpy`, development dependencies for `pytest`, `pytest-cov`, and `ruff`, and a `campusvision = "campusvision.app:main"` console script. Configure pytest with `pythonpath = ["src"]` and Ruff for Python 3.11. Add `.venv/`, model caches, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, and generated `outputs/` to `.gitignore`.

Write tests that assert the registry exposes all ten user-facing tasks, maps Object Detection to `<OD>`, maps Referring Expression Segmentation to `<REFERRING_EXPRESSION_SEGMENTATION>`, returns `"<OPEN_VOCABULARY_DETECTION>bicycle"` for trimmed text, rejects empty phrase text with `TaskInputError`, ignores stale phrase text for `<CAPTION>`, and rejects unknown task names.

- [ ] **Step 2: Run the focused test and confirm the contracts do not exist yet**

Run: `python -m pytest tests/test_tasks.py -q`

Expected: collection fails because `campusvision.tasks` does not exist.

- [ ] **Step 3: Implement immutable task definitions and validation**

Define:

```python
@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    token: str
    requires_text: bool
    result_kind: Literal["text", "boxes", "ocr", "polygons", "mixed"]


class TaskInputError(ValueError):
    pass
```

Populate `TASKS` with the ten tasks in the Global Constraints. `build_prompt` must trim text, enforce a 500-character maximum, concatenate the Florence task token directly with required text, and return only the task token for tasks without text input.

- [ ] **Step 4: Verify and lint the foundation**

Run: `python -m pytest tests/test_tasks.py -q`

Expected: all task tests pass.

Run: `python -m ruff check src/campusvision/contracts.py src/campusvision/tasks.py tests/test_tasks.py`

Expected: no lint errors.

- [ ] **Step 5: Commit the task contracts**

```bash
git add pyproject.toml .gitignore src/campusvision/__init__.py src/campusvision/contracts.py src/campusvision/tasks.py tests/test_tasks.py
git commit -m "feat: define CampusVision task contracts"
```

### Task 2: Image normalization and spatial rendering

**Files:**
- Create: `src/campusvision/imaging.py`
- Create: `tests/test_imaging.py`

**Interfaces:**
- Consumes: A `PIL.Image.Image` and the parsed Florence result `dict[str, object]`.
- Produces: `normalize_image(image: Image.Image, max_side: int = 1600) -> NormalizedImage`, `render_result(image: Image.Image, task: TaskSpec, parsed: dict[str, object]) -> Image.Image`, and `NormalizedImage(original_size, processed, resized)`.

- [ ] **Step 1: Write failing normalization and rendering tests**

Create synthetic fixtures in memory and assert that normalization applies EXIF orientation, converts palette/RGBA/grayscale inputs to RGB, preserves aspect ratio, never enlarges an image, and limits the longest side to 1600. Assert that rendering leaves the input bytes unchanged, clamps boxes to the canvas, assigns a fallback label when labels are missing, draws OCR quadrilaterals from `quad_boxes`, and fills nested segmentation polygons with a translucent overlay. Include an empty-result test that returns an unchanged copy.

- [ ] **Step 2: Confirm the imaging module is absent**

Run: `python -m pytest tests/test_imaging.py -q`

Expected: collection fails because `campusvision.imaging` does not exist.

- [ ] **Step 3: Implement deterministic normalization and rendering**

Use `ImageOps.exif_transpose`, `Image.Resampling.LANCZOS`, and `ImageDraw`. Keep normalization and rendering separate. Recursively normalize Florence polygon nesting into point lists, reject nonnumeric and nonfinite coordinates, clamp all coordinates to image bounds, choose colors from a fixed accessible palette based on label index, draw labels on filled backgrounds, and render on a copy so the source object is never changed.

- [ ] **Step 4: Verify rendering behavior**

Run: `python -m pytest tests/test_imaging.py -q`

Expected: all imaging tests pass.

Run: `python -m ruff check src/campusvision/imaging.py tests/test_imaging.py`

Expected: no lint errors.

- [ ] **Step 5: Commit image handling**

```bash
git add src/campusvision/imaging.py tests/test_imaging.py
git commit -m "feat: render Florence spatial outputs"
```

### Task 3: Run artifacts and measurements

**Files:**
- Create: `src/campusvision/artifacts.py`
- Create: `src/campusvision/metrics.py`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: A completed or failed inference run and optional annotated `PIL.Image.Image`.
- Produces: `RunRecord`, `RunArtifacts`, `ArtifactStore.save_success(...) -> RunArtifacts`, `ArtifactStore.save_failure(...) -> RunArtifacts`, `rss_megabytes() -> float`, and `synchronize_device(device_type: str) -> None`.

- [ ] **Step 1: Write failing persistence and metrics tests**

Use `tmp_path` to assert that two run IDs create separate directories, successful runs save UTF-8 JSON and PNG, failures save JSON without PNG, and concurrent threads append complete CSV rows under a fixed column order. Assert JSON handles tuples and NumPy scalar values. Patch psutil to verify byte-to-MiB conversion. Patch torch backends to verify CUDA and MPS synchronization calls and CPU no-op behavior.

- [ ] **Step 2: Confirm persistence and metric imports fail**

Run: `python -m pytest tests/test_artifacts.py tests/test_metrics.py -q`

Expected: collection fails because the modules do not exist.

- [ ] **Step 3: Implement typed run records and atomic artifact writes**

Define `RunRecord` with run ID, UTC timestamp, status, error, checkpoint, revision, library versions, device, dtype, task, phrase, original and processed sizes, resized flag, generation settings, cold-start flag, model-load seconds, inference seconds, total seconds, RSS before/after, accelerator memory, and output paths. Use UUID-based run directories. Write JSON and PNG to sibling temporary files and replace their final paths only after successful writes. Guard CSV header detection and append with one process-local lock and `newline=""`.

Implement a JSON conversion helper for dataclasses, `Path`, tuple, NumPy values, and nested mappings/lists. `synchronize_device` calls `torch.cuda.synchronize()` for CUDA, `torch.mps.synchronize()` for MPS when available, and returns immediately for CPU.

- [ ] **Step 4: Verify persistence, concurrency, and measurements**

Run: `python -m pytest tests/test_artifacts.py tests/test_metrics.py -q`

Expected: all artifact and metric tests pass, including the concurrent writer test.

- [ ] **Step 5: Commit artifact persistence**

```bash
git add src/campusvision/artifacts.py src/campusvision/metrics.py tests/test_artifacts.py tests/test_metrics.py
git commit -m "feat: persist reproducible inference runs"
```

### Task 4: Device policy and Florence inference service

**Files:**
- Create: `src/campusvision/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `build_prompt`, `normalize_image`, `render_result`, `ArtifactStore`, and a `ModelBackend` protocol.
- Produces: `DeviceChoice`, `choose_device(requested: str = "auto") -> DeviceChoice`, `FlorenceBackend.load(...)`, `FlorenceBackend.generate(...)`, `InferenceService.run(image, task_name, text_input, max_new_tokens, num_beams) -> InferenceOutcome`, and `get_inference_service() -> InferenceService`.

- [ ] **Step 1: Write failing device and service tests with a controlled backend**

Assert auto-selection order CUDA then MPS then CPU, CPU uses float32, unsupported explicit devices raise `DeviceUnavailableError`, and explicit unavailable CUDA does not silently fall back. Use a fake backend to assert input validation happens before `load`, two sequential runs load once, concurrent runs never overlap `generate`, task token and phrase reach the backend exactly, processor post-processing receives processed image dimensions, success writes artifacts, and a backend exception writes a failed record then raises `InferenceError` with recovery context. Assert `max_new_tokens` is bounded to 16–2048 and `num_beams` to 1–5.

- [ ] **Step 2: Confirm inference contracts are absent**

Run: `python -m pytest tests/test_inference.py -q`

Expected: collection fails because `campusvision.inference` does not exist.

- [ ] **Step 3: Implement device selection and lazy serialized inference**

Create `DeviceChoice(device: torch.device, dtype: torch.dtype, label: str)` and explicit `DeviceUnavailableError`/`InferenceError`. Use one load lock and one inference lock. Validate prompt and generation settings before model loading. Measure preprocessing, synchronized generation, decoding/post-processing, rendering, and total time separately enough to calculate the recorded model-load, inference, and total fields.

Implement `FlorenceBackend` using `AutoProcessor.from_pretrained` and `AutoModelForCausalLM.from_pretrained` with the named checkpoint, pinned revision constant, selected dtype, `trust_remote_code=True`, `local_files_only=False`, `model.eval()`, and `torch.inference_mode()`. Move `input_ids` and `pixel_values` to the chosen device without casting token IDs. Decode with special tokens retained, then call `processor.post_process_generation(generated_text, task=task.token, image_size=image.size)`.

- [ ] **Step 4: Verify service behavior without downloading model weights**

Run: `python -m pytest tests/test_inference.py -q`

Expected: all device, serialization, measurement, and error tests pass using the controlled backend.

Run: `python -m pytest -q`

Expected: the full unit suite passes without network access or model loading.

- [ ] **Step 5: Commit inference integration**

```bash
git add src/campusvision/inference.py tests/test_inference.py
git commit -m "feat: integrate Florence inference service"
```

### Task 5: Gradio application and campus presets

**Files:**
- Create: `src/campusvision/app.py`
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `TASKS`, `InferenceService.run`, `InferenceOutcome`, and saved artifact paths.
- Produces: `run_from_ui(image, task_name, text_input, max_new_tokens, num_beams) -> tuple[Image.Image | None, str, dict[str, object], str | None, str | None, str]`, `build_app(service: InferenceService | None = None) -> gr.Blocks`, and `main() -> None`.

- [ ] **Step 1: Write failing UI adapter tests**

With a fake service, assert the adapter returns annotated image, generated text, structured JSON, PNG path, JSON path, and a metrics summary. Assert missing images and invalid phrase input become concise user-facing `gr.Error` failures. Inspect the built Blocks configuration or registered components to verify image upload/webcam/clipboard sources, all ten tasks, phrase textbox, generation controls, output image, text, JSON, two file downloads, status, and campus example presets are present.

- [ ] **Step 2: Confirm the app module is absent**

Run: `python -m pytest tests/test_app.py -q`

Expected: collection fails because `campusvision.app` does not exist.

- [ ] **Step 3: Build the thin Gradio interface**

Use `gr.Blocks` with clear help text that identifies the pretrained checkpoint and limitations. Configure `gr.Image(type="pil", sources=["upload", "webcam", "clipboard"])`, a dropdown populated from `TASKS`, an optional phrase box that is visibly required for phrase tasks, bounded sliders for tokens and beams, and a Run button. Show annotated output, generated text, parsed JSON, runtime/device summary, and separate downloadable PNG/JSON files. Include presets for a classroom caption, noticeboard OCR, person/object grounding, and campus-object open-vocabulary detection; presets set controls but supply no fabricated images.

Call `demo.queue(default_concurrency_limit=1)` and `demo.launch(share=False)` in `main`. Do not launch a server at import time.

- [ ] **Step 4: Verify UI wiring and the whole unit suite**

Run: `python -m pytest tests/test_app.py -q`

Expected: all adapter and component tests pass.

Run: `python -m pytest --cov=campusvision --cov-report=term-missing -q`

Expected: all tests pass and coverage identifies no untested critical validation branch.

- [ ] **Step 5: Commit the application UI**

```bash
git add src/campusvision/app.py tests/test_app.py
git commit -m "feat: add CampusVision Gradio interface"
```

### Task 6: Reproducible setup, real-model smoke test, and evaluation guide

**Files:**
- Create: `README.md`
- Create: `scripts/smoke_test.py`
- Create: `tests/fixtures/campus_notice.png`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Installed application, `get_inference_service`, and a redistributable synthetic fixture containing visible campus-notice text and simple objects.
- Produces: A documented installation workflow, a command-line real-model smoke test, and a dependency lock or exact installed-version record generated by the selected environment tool.

- [ ] **Step 1: Create a deterministic local smoke fixture and smoke-test command**

Add a small repository-owned synthetic image labelled `CAMPUS LIBRARY`, with a door, noticeboard, and directional arrow. Implement `scripts/smoke_test.py` with `--task`, `--text`, `--image`, `--device`, and `--output-dir` arguments. It must invoke the real service, print generated text, parsed JSON, device/dtype, model-load time, inference time, RSS delta, and artifact paths, and exit nonzero on model or rendering failure.

- [ ] **Step 2: Document setup and evaluation precisely**

Write Windows PowerShell and macOS shell instructions for installing Python 3.11, creating `.venv`, installing the project, running tests, launching `campusvision`, and running the smoke command. Explain the first-run model download and approximate cache behavior without promising a download size. State that Python 3.14 is outside the supported project range.

Document every task and required phrase behavior, output directory layout, CSV field meanings, cold versus warm timing, accelerator synchronization, RSS limitations, how to repeat each task at least five warm times, and how to compare machines using identical checkpoint revision, dependency lock, input file, processed dimensions, task, phrase, token limit, and beam count. Explain that accuracy requires labelled ground truth and that the included synthetic fixture is a smoke check, not an accuracy benchmark.

- [ ] **Step 3: Install, resolve, and record a validated dependency set**

Create a Python 3.11 virtual environment. Install the editable project and development dependencies, resolve the checkpoint with the selected Transformers version, and generate the environment tool's exact lock file or a sorted `requirements-lock.txt` from the validated environment. Replace broad dependency ranges in `pyproject.toml` with the compatible ranges proven by this run while keeping platform-appropriate PyTorch installation instructions in the README.

- [ ] **Step 4: Run static checks and the complete unit suite**

Run: `python -m ruff check .`

Expected: no lint errors.

Run: `python -m pytest -q`

Expected: all unit tests pass without requiring model download.

- [ ] **Step 5: Run real Florence-2 smoke checks**

Run caption, object detection, OCR with regions, phrase grounding for `the noticeboard`, referring-expression segmentation for `the door`, and open-vocabulary detection for `directional arrow` against the fixture. Reopen every emitted PNG with Pillow, assert `verify()` succeeds, confirm its dimensions match the processed image, and inspect the spatial images for visible, in-bounds annotations. Record which experimental tasks return empty or weak results without altering them into successes.

Expected: caption, detection, and OCR commands complete with real model output and valid artifacts; phrase and polygon task outcomes are recorded accurately. Any unsupported task remains visible as a limitation with the exact error or empty output.

- [ ] **Step 6: Launch and probe the local interface**

Run `campusvision`, open the local URL, verify the task controls and fixture upload, execute one caption task, download its JSON, and stop the server cleanly. Confirm no public share URL is created.

- [ ] **Step 7: Commit documentation and verified dependency record**

```bash
git add README.md scripts/smoke_test.py tests/fixtures/campus_notice.png pyproject.toml requirements-lock.txt
git commit -m "docs: add reproducible CampusVision workflow"
```

### Task 7: Final verification and review handoff

**Files:**
- Modify only files implicated by verification failures.

**Interfaces:**
- Consumes: The complete CampusVision repository.
- Produces: A clean, tested branch and a concise verification record in the final handoff.

- [ ] **Step 1: Run the full validation set from a clean process**

Run `python -m ruff check .`, `python -m pytest --cov=campusvision --cov-report=term-missing -q`, and the real caption smoke test. Confirm `git diff --check` is clean and `git status --short` contains only intended files.

- [ ] **Step 2: Re-check the approved design against the delivered behavior**

Confirm all ten tasks are selectable, phrase validation precedes loading, the actual checkpoint and revision are logged, device override behavior is explicit, artifacts are unique, CSV rows are complete, downloads work, and limitations are present in the UI and README. Confirm tests using a fake backend are identified as unit tests and the actual smoke result is identified separately.

- [ ] **Step 3: Request whole-branch code review**

Use the required review workflow to inspect correctness, security of remote model code/revision pinning, concurrent behavior, cross-platform paths, resource bounds, and user-facing claims. Fix verified issues and rerun the affected checks plus the full unit suite.

- [ ] **Step 4: Commit review fixes if any**

If review produced changes, inspect `git status --short`, stage each listed CampusVision file by its exact path, and run `git commit -m "fix: address CampusVision review findings"`. Skip this step when review required no changes.

- [ ] **Step 5: Prepare the handoff**

Report the local launch command, Python version, checkpoint/revision, actual tested device, unit-test result, real smoke tasks exercised, sample output paths, known model limitations, and any hardware path that could not be tested on this machine.
