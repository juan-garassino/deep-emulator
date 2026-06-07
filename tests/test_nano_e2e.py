"""Nano end-to-end pipeline test.

Runs the full collect→pretrain→RL→eval→viz loop on a synthetic env with
parameters tuned for ~30 seconds of test wall-time. Verifies every stage
produces the expected artifacts.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Load the script module from disk (it's under scripts/, not in the package)
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "nano_e2e.py"
_spec = importlib.util.spec_from_file_location("nano_e2e", _SCRIPT)
_nano = importlib.util.module_from_spec(_spec)
sys.modules["nano_e2e"] = _nano
_spec.loader.exec_module(_nano)


@pytest.mark.timeout(300)
def test_nano_pipeline_produces_all_artifacts(tmp_path):
    artifacts = _nano.run_nano_pipeline(
        tmp_path,
        n_frames=128,
        dino_iters=5,
        rl_steps=20,
        eval_episodes=2,
        gif_steps=4,
        verbose=False,
    )

    # 1. Frame corpus
    assert artifacts["corpus_dir"].exists()
    assert (artifacts["corpus_dir"] / "index.json").exists()
    assert any((artifacts["corpus_dir"]).glob("chunk_*.npz"))

    # 2. DINO encoder bundle (full + encoder-only + metadata + latest.txt)
    enc = artifacts["encoder_dir"]
    assert (enc / "encoder.pt").exists()
    assert (enc / "encoder_only.pt").exists()
    enc_md = json.loads((enc / "metadata.json").read_text())
    assert enc_md["algo"] == "dino"
    assert enc_md["step_count"] == 5
    assert (enc.parent / "latest.txt").read_text().strip() == str(enc.resolve())

    # 3. RL bundle records the encoder linkage in metadata
    rl = artifacts["rl_bundle"]
    rl_md = json.loads((rl / "metadata.json").read_text())
    assert rl_md["algo"] == "ddqn"
    assert rl_md["observation_shape"] == [3 * 64]  # frame_stack * latent_dim
    assert "encoder" in rl_md
    assert rl_md["encoder"]["frozen"] is True
    assert rl_md["encoder"]["latent_dim"] == 64
    assert Path(rl_md["encoder"]["path"]).resolve() == enc.resolve()
    assert (rl / "model.pt").exists()
    assert (rl / "metrics.tsv").exists()

    # 4. Eval artifacts
    assert (rl / "eval_episodes.tsv").exists()
    assert artifacts["eval_report"].exists()
    html = artifacts["eval_report"].read_text()
    assert "SYNTH BLOB" in html
    assert "training reward curves" in html.lower() or "reward" in html.lower()

    # 5. Attention GIF (only if PIL available)
    if importlib.util.find_spec("PIL") is not None:
        assert artifacts["attention_gif"] is not None
        assert artifacts["attention_gif"].exists()
        assert artifacts["attention_gif"].stat().st_size > 1024  # non-trivial

    # 6. Pipeline sanity
    assert isinstance(artifacts["dino_first_loss"], float)
    assert isinstance(artifacts["dino_last_loss"], float)
    assert artifacts["rl_learning_steps"] >= 0  # may be 0 if all steps were in burnin
    assert isinstance(artifacts["eval_mean_reward"], float)


def test_nano_script_main_runs(tmp_path):
    """The CLI entrypoint should run with tiny params and exit 0."""
    sys.argv = [
        "nano_e2e.py",
        "--out", str(tmp_path),
        "--frames", "64",
        "--dino-iters", "3",
        "--rl-steps", "10",
        "--eval-episodes", "1",
        "--gif-steps", "2",
    ]
    rc = _nano.main()
    assert rc == 0
    assert (tmp_path / "eval_report.html").exists()
