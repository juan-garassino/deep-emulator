"""Sanity tests for the Colab wrapper — runs outside Colab."""
from __future__ import annotations

from pathlib import Path


def test_module_imports():
    from deepEmulator.training import colab_train

    assert hasattr(colab_train, "run_in_colab")
    assert hasattr(colab_train, "main")


def test_in_colab_false_outside_colab():
    from deepEmulator.training.colab_train import _in_colab

    assert _in_colab() is False


def test_resolve_passthrough_when_no_drive(tmp_path):
    from deepEmulator.training.colab_train import _resolve

    p = tmp_path / "foo.gb"
    assert _resolve(p, None) == p
    # absolute paths bypass drive_root even if provided
    assert _resolve("/abs/x", tmp_path) == Path("/abs/x")


def test_resolve_joins_drive_root_for_relative(tmp_path):
    from deepEmulator.training.colab_train import _resolve

    assert _resolve("deepEmulator/roms/PokemonRed.gb", tmp_path) == (
        tmp_path / "deepEmulator/roms/PokemonRed.gb"
    )


def test_notebook_is_valid_json():
    import json

    nb = json.loads((Path(__file__).resolve().parent.parent / "notebooks" / "01_colab_train.ipynb").read_text())
    assert nb["nbformat"] == 4
    assert any("run_in_colab" in "".join(cell.get("source", [])) for cell in nb["cells"])
