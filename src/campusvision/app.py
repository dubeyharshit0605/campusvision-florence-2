from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from campusvision.contracts import TaskInputError
from campusvision.imaging import clean_label
from campusvision.inference import (
    DeviceUnavailableError,
    InferenceError,
    InferenceService,
    get_inference_service,
)
from campusvision.tasks import TASKS


def _status(record: Any) -> str:
    rss_change = record.rss_after_mb - record.rss_before_mb
    cold = "cold start" if record.cold_start else "warm run"
    return (
        f"Device: {record.device.upper()} ({record.dtype}) | {cold} | "
        f"Load: {record.model_load_seconds:.3f} s | "
        f"Inference: {record.inference_seconds:.3f} s | "
        f"Total: {record.total_seconds:.3f} s | "
        f"Process RSS change: {rss_change:+.1f} MiB"
    )


def run_from_ui(
    image: Image.Image | None,
    task_name: str,
    text_input: str,
    max_new_tokens: int,
    num_beams: int,
    *,
    service: InferenceService | None = None,
) -> tuple[Image.Image | None, str, dict[str, object], str | None, str | None, str]:
    if image is None:
        raise gr.Error("Upload, paste, or capture an image before running a task.")
    selected_service = service or get_inference_service()
    try:
        outcome = selected_service.run(
            image,
            task_name,
            text_input,
            int(max_new_tokens),
            int(num_beams),
        )
    except (TaskInputError, DeviceUnavailableError, InferenceError) as exc:
        raise gr.Error(str(exc)) from exc

    image_path = outcome.artifacts.image_path
    return (
        outcome.annotated_image,
        clean_label(outcome.generated_text),
        outcome.parsed,
        str(image_path) if image_path else None,
        str(outcome.artifacts.json_path),
        _status(outcome.record),
    )


def build_app(service: InferenceService | None = None) -> gr.Blocks:
    with gr.Blocks(title="CampusVision — Florence-2") as demo:
        gr.Markdown(
            """
            # CampusVision
            Run Microsoft `Florence-2-base-ft` locally for captioning, detection, OCR,
            grounding, and segmentation. Results are model predictions and can be wrong;
            use labelled data for accuracy evaluation.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Campus image",
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                )
                task_input = gr.Dropdown(
                    label="Vision task",
                    choices=list(TASKS),
                    value="Caption",
                )
                phrase_input = gr.Textbox(
                    label="Phrase or object to locate",
                    placeholder=(
                        "Required for grounding, segmentation, and open-vocabulary detection"
                    ),
                    max_lines=3,
                )
                with gr.Row():
                    max_tokens = gr.Slider(
                        16,
                        2048,
                        value=512,
                        step=16,
                        label="Maximum generated tokens",
                    )
                    beams = gr.Slider(1, 5, value=3, step=1, label="Beam count")
                run_button = gr.Button("Run Florence-2", variant="primary")
            with gr.Column(scale=1):
                annotated_output = gr.Image(label="Annotated result", type="pil")
                generated_output = gr.Textbox(label="Generated text", lines=4)
                structured_output = gr.JSON(label="Structured result")
                status_output = gr.Textbox(label="Run details", interactive=False)
                with gr.Row():
                    image_download = gr.File(label="Download annotated PNG")
                    json_download = gr.File(label="Download result JSON")

        gr.Examples(
            examples=[
                ["Detailed Caption", ""],
                ["OCR with Regions", ""],
                ["Phrase Grounding", "the student holding a laptop"],
                ["Open Vocabulary Detection", "noticeboard"],
            ],
            inputs=[task_input, phrase_input],
            label="Campus task presets",
        )

        def handler(image, task, phrase, tokens, beam_count):
            return run_from_ui(
                image,
                task,
                phrase,
                tokens,
                beam_count,
                service=service,
            )

        run_button.click(
            handler,
            inputs=[image_input, task_input, phrase_input, max_tokens, beams],
            outputs=[
                annotated_output,
                generated_output,
                structured_output,
                image_download,
                json_download,
                status_output,
            ],
        )
    return demo


def main() -> None:
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(share=False)


if __name__ == "__main__":
    main()
