"""Hugging Face Spaces entry point."""

from campusvision.app import build_app

demo = build_app()
demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(share=False)
