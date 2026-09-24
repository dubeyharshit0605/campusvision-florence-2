"""Deploy CampusVision as a scale-to-zero Modal web application."""

from __future__ import annotations

import os
import subprocess
import sys

import modal

APP_NAME = "campusvision-florence-2"
PORT = 8000

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("campusvision-model-cache", create_if_missing=True)
run_data = modal.Volume.from_name("campusvision-run-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "einops==0.8.2",
        "gradio==5.49.1",
        "numpy==2.4.6",
        "pillow==11.3.0",
        "psutil==7.2.2",
        "timm==1.0.30",
        "transformers==4.49.0",
    )
    .add_local_dir("src", remote_path="/root/src", copy=True)
)


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=1800,
    startup_timeout=1200,
    scaledown_window=300,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/data": run_data,
    },
    env={
        "PYTHONPATH": "/root/src",
        "PORT": str(PORT),
        "CAMPUSVISION_OUTPUT_DIR": "/data/outputs",
    },
)
@modal.concurrent(max_inputs=10)
@modal.web_server(PORT, startup_timeout=1200)
def serve() -> None:
    """Start the Gradio server inside the Modal container."""
    subprocess.Popen(
        [sys.executable, "-m", "campusvision.modal_entrypoint"],
        env=os.environ.copy(),
    )
