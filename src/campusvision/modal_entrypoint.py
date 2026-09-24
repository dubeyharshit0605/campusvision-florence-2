"""Container entry point used by the Modal deployment."""

from __future__ import annotations

import os

from campusvision.app import build_app
from campusvision.inference import get_inference_service


def main() -> None:
    """Start Gradio on Modal's externally routed container port."""
    port = int(os.environ.get("PORT", "8000"))
    output_dir = os.environ.get("CAMPUSVISION_OUTPUT_DIR", "/data/outputs")
    demo = build_app(service=get_inference_service(output_dir=output_dir))
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
