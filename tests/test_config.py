from pathlib import Path

import pytest

from autonomis.config import load_config


def test_default_config_allows_explicit_training_actions():
    config = load_config("configs/campaign.yaml")
    assert config.dataset.is_file()
    assert config.dataset_role == "public_reference"
    assert config.backend == "cpu"
    config.permit.check()


def test_gpu_config_is_valid_without_host_binding(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    original = Path("configs/campaign.yaml").read_text()
    path = folder / "campaign.yaml"
    path.write_text(original.replace("backend: cpu", "backend: gpu"))
    config = load_config(path)
    assert config.backend == "gpu"
    config.permit.check()


@pytest.mark.parametrize("change", ["budget: 16", "backend: tpu", "threads: 0",
    "enable_training: 'false'", "output_dir: ../../outside", "unexpected: true"])
def test_unsafe_configuration_is_rejected(tmp_path, change):
    folder = tmp_path / "configs"
    folder.mkdir()
    original = Path("configs/campaign.yaml").read_text()
    key = change.split(":", 1)[0]
    lines = [line for line in original.splitlines() if not line.startswith(f"{key}:")]
    path = folder / "campaign.yaml"
    path.write_text("\n".join([*lines, change]))
    with pytest.raises(ValueError):
        load_config(path)
