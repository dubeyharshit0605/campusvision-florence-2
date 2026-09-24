from unittest.mock import Mock

from campusvision import metrics


def test_rss_megabytes_converts_bytes(monkeypatch) -> None:
    process = Mock()
    process.memory_info.return_value.rss = 10 * 1024 * 1024
    monkeypatch.setattr(metrics.psutil, "Process", Mock(return_value=process))

    assert metrics.rss_megabytes() == 10.0


def test_cpu_synchronization_is_a_noop(monkeypatch) -> None:
    cuda_sync = Mock()
    mps_sync = Mock()
    monkeypatch.setattr(metrics.torch.cuda, "synchronize", cuda_sync)
    monkeypatch.setattr(metrics.torch.mps, "synchronize", mps_sync)

    metrics.synchronize_device("cpu")

    cuda_sync.assert_not_called()
    mps_sync.assert_not_called()


def test_accelerator_synchronization_uses_matching_backend(monkeypatch) -> None:
    cuda_sync = Mock()
    mps_sync = Mock()
    monkeypatch.setattr(metrics.torch.cuda, "synchronize", cuda_sync)
    monkeypatch.setattr(metrics.torch.backends.mps, "is_available", Mock(return_value=True))
    monkeypatch.setattr(metrics.torch.mps, "synchronize", mps_sync)

    metrics.synchronize_device("cuda")
    metrics.synchronize_device("mps")

    cuda_sync.assert_called_once_with()
    mps_sync.assert_called_once_with()
