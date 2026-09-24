# CampusVision implementation design

## Intended outcome

Build the CampusVision application described in the supplied Florence-2 paper breakdown in this repository. The user confirmed using the currently empty AI PROJECT folder. Success means a runnable local application that performs real pretrained-model inference, displays spatial results, exports outputs, and records reproducible measurements. This is application integration and evaluation, not reproduction of Florence-2 training.

## Approach

Use Python, PyTorch, Hugging Face Transformers and Gradio with microsoft/Florence-2-base-ft. This directly matches the supplied brief and keeps model execution local after the initial download. A separate API and JavaScript frontend would add deployment complexity without serving the current brief. Training an architecture from scratch would require a separate dataset, compute budget and evaluation scope.

Use an isolated environment and pin a dependency combination validated with this checkpoint. Prefer the native Transformers implementation when checkpoint compatibility is verified. If the checkpoint requires repository-provided Python code, inspect that code and pin its revision. Do not rely on an untested latest-version dependency combination. The registered Python runtime is currently 3.14; use a compatible isolated Python runtime if required by dependencies.

## User experience

The application runs on localhost. A single screen provides image upload, webcam capture and clipboard input, a task selector, optional phrase input, an image preview, a run button, a results panel and downloads. Webcam support depends on browser permission and available hardware. The app does not automatically enable public sharing.

Tasks include caption, detailed caption, more detailed caption, object detection, OCR, OCR with regions, dense region captioning, phrase grounding, referring-expression segmentation and open-vocabulary detection. Phrase-based tasks require nonempty text. Campus presets supply task and phrase examples without pretending to provide real campus photographs or ground-truth annotations.

Results show generated text, parsed structured data, and an annotated image when spatial data exists. Downloads include PNG and JSON; experiment history is exportable as CSV. Experimental tasks remain labelled as such if the checkpoint cannot demonstrate reliable behavior during verification.

## Architecture and data flow

- A task registry owns prompt tokens, input requirements and output categories.
- An inference service owns model loading, device selection, preprocessing, generation and the model processor's output parsing.
- A rendering module handles boxes, OCR quadrilaterals and nested segmentation polygons without changing the source image.
- An experiment store records per-run metadata and writes unique output files.
- A Gradio interface connects these components and presents validation and runtime errors.

Flow: normalize image orientation and RGB mode, validate task and phrase, load the cached model, preprocess with the checkpoint processor, generate under inference mode, decode without discarding location tokens, post-process using the dimensions of the image being annotated, render, save and display.

Optional input resizing preserves aspect ratio. All coordinates and output dimensions refer to the actual image used for inference and rendering. Record both original and processed dimensions. The processor may internally resize images to a fixed size; changing source-image resolution is not presented as changing the model's visual-token count.

## Runtime and resource behavior

Select CUDA when available, otherwise MPS when available, otherwise CPU. Use a supported dtype per device, with float32 as the CPU default. Provide an explicit device override and report the actual device used. Device failures must produce actionable errors; any fallback is visible in results and logs.

Cache a single model instance and serialize inference to avoid concurrent model copies and interleaved measurements. Bound image dimensions, phrase length and generated token count. Load the model on first inference and show initialization status. Distinguish initial download/loading from warm inference.

The current machine reports approximately 16 GiB RAM. Accelerator availability still needs runtime verification. Performance numbers must come from actual measurements; no Mac comparison or accuracy claim is inferred from this Windows environment.

## Measurements and persistence

Record a run identifier, timestamp, checkpoint and revision, software versions, device, dtype, task, phrase, original and processed image dimensions, generation settings, model-load time, inference time, total processing time, process RSS before/after, applicable accelerator-memory statistics, status and output paths.

Synchronize accelerators around timed inference. Describe process RSS as a process-memory observation, not isolated model memory or peak memory. Separate warm runs from cold starts. Save PNG/JSON in unique per-run directories under outputs; use a consistent CSV schema and serialize writes. Failed runs do not masquerade as successful measurements.

## Errors and limitations

Validate missing or invalid images, required phrases and unsupported task names before inference. Surface download, dependency, device, memory and generation errors with concrete recovery guidance. Preserve parsed JSON even when rendering cannot represent an output; report rendering failures visibly. Never return mock predictions as model results.

Captions and OCR can be incorrect, object detection can miss objects, and polygons are generated outlines rather than guaranteed pixel-accurate masks. Accuracy evaluation requires labelled examples supplied or collected separately. No training, user accounts, cloud deployment or campus surveillance pipeline is included.

## Verification and deliverables

Test task validation, prompt construction, orientation and coordinate consistency, box/quadrilateral/polygon rendering, JSON serialization, CSV persistence and inference-service integration using a controlled model substitute. These tests verify application behavior and must be reported separately from real inference.

Run real checkpoint smoke tests on available images for captioning, detection and OCR, and exercise phrase and polygon tasks where feasible. Inspect saved annotated output and verify the interface starts and exposes expected controls. Record failures and untested hardware explicitly. Supply README setup/run instructions for Windows and macOS, pinned dependencies, application source, tests and an evaluation guide describing repeatable warm/cold runs and labelled-data requirements.

## References

- User-provided Florence-2 Simple Paper Breakdown, including the CampusVision feature list.
- https://huggingface.co/microsoft/Florence-2-base-ft
- https://huggingface.co/microsoft/Florence-2-base-ft/blob/main/processing_florence2.py
- https://huggingface.co/docs/transformers/model_doc/florence2

## Review status

Written design prepared for user review. Application code and dependencies have not yet been created or installed.
