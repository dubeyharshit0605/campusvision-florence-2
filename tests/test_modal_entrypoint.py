from campusvision import modal_entrypoint


class FakeDemo:
    def __init__(self) -> None:
        self.queue_limit = None
        self.launch_options = None

    def queue(self, *, default_concurrency_limit: int) -> None:
        self.queue_limit = default_concurrency_limit

    def launch(self, **options) -> None:
        self.launch_options = options


def test_modal_entrypoint_binds_public_container_port(monkeypatch, tmp_path):
    demo = FakeDemo()
    captured = {}

    monkeypatch.setenv("PORT", "8123")
    monkeypatch.setenv("CAMPUSVISION_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        modal_entrypoint,
        "get_inference_service",
        lambda output_dir: captured.setdefault("output_dir", output_dir),
    )
    monkeypatch.setattr(
        modal_entrypoint,
        "build_app",
        lambda service: captured.setdefault("service", service) or demo,
    )

    # Return the demo explicitly because the service sentinel above is truthy.
    monkeypatch.setattr(modal_entrypoint, "build_app", lambda service: demo)
    modal_entrypoint.main()

    assert captured["output_dir"] == str(tmp_path)
    assert demo.queue_limit == 1
    assert demo.launch_options == {
        "server_name": "0.0.0.0",
        "server_port": 8123,
        "share": False,
    }
