"""Tests for the real reseed op (LoRA adapter perturbation).

Runs on CPU against a synthetic adapter dir — no base model, no GPU.
"""

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from si.phase2_ops import perturb_lora_adapter


def _make_adapter(d, tensors, *, config_text='{"r": 8}'):
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
    (d / "adapter_config.json").write_text(config_text)


def test_perturb_adds_noise_preserves_shapes_and_copies_config(tmp_path):
    parent = tmp_path / "parent"
    dest = tmp_path / "dest"
    base = {
        "lora_A": torch.zeros(4, 8),
        "lora_B": torch.ones(8, 4),
    }
    _make_adapter(parent, base)

    gen = torch.Generator().manual_seed(0)
    perturb_lora_adapter(str(parent), str(dest), 0.01, generator=gen)

    assert (dest / "adapter_model.safetensors").exists()
    assert (dest / "adapter_config.json").read_text() == '{"r": 8}'

    with safe_open(str(dest / "adapter_model.safetensors"), framework="pt") as f:
        assert f.metadata() == {"format": "pt"}
        a = f.get_tensor("lora_A")
        b = f.get_tensor("lora_B")
    # shapes preserved
    assert a.shape == (4, 8) and b.shape == (8, 4)
    # weights moved off their exact init values...
    assert not torch.equal(a, base["lora_A"])
    assert not torch.equal(b, base["lora_B"])
    # ...by a small amount consistent with sigma=0.01 (not a large displacement)
    assert (a - base["lora_A"]).abs().mean() < 0.05
    assert (b - base["lora_B"]).abs().mean() < 0.05


def test_perturb_is_deterministic_under_seed(tmp_path):
    parent = tmp_path / "p"
    _make_adapter(parent, {"w": torch.ones(16, 16)})

    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    perturb_lora_adapter(str(parent), str(d1), 0.02, generator=torch.Generator().manual_seed(7))
    perturb_lora_adapter(str(parent), str(d2), 0.02, generator=torch.Generator().manual_seed(7))

    with safe_open(str(d1 / "adapter_model.safetensors"), framework="pt") as f:
        w1 = f.get_tensor("w")
    with safe_open(str(d2 / "adapter_model.safetensors"), framework="pt") as f:
        w2 = f.get_tensor("w")
    assert torch.equal(w1, w2)


def test_perturb_leaves_integer_tensors_untouched(tmp_path):
    parent = tmp_path / "p"
    ints = torch.arange(10, dtype=torch.int64)
    _make_adapter(parent, {"idx": ints, "w": torch.ones(3, 3)})
    dest = tmp_path / "d"
    perturb_lora_adapter(str(parent), str(dest), 0.1, generator=torch.Generator().manual_seed(1))
    with safe_open(str(dest / "adapter_model.safetensors"), framework="pt") as f:
        assert torch.equal(f.get_tensor("idx"), ints)


def test_missing_weights_raises(tmp_path):
    parent = tmp_path / "empty"
    parent.mkdir()
    import pytest

    with pytest.raises(FileNotFoundError):
        perturb_lora_adapter(str(parent), str(tmp_path / "d"), 0.01)
