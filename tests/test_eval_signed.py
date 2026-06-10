"""eval_signed round-trip — would have caught the NaN-score bug (the old
parser grepped the HTML for a marker deepemu-eval never emitted)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("eval_signed", REPO / "scripts" / "eval_signed.py")
eval_signed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_signed)


def _fake_bundles(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "baseline"
    treat = tmp_path / "treatment"
    for b in (base, treat):
        b.mkdir()
        (b / "model.pt").write_bytes(b"weights" + b.name.encode())
        (b / "eval_episodes.tsv").write_text("episode\treward\tlength\n0\t1.0\t10\n")
    return base, treat


def test_signed_row_roundtrip(tmp_path, monkeypatch):
    base, treat = _fake_bundles(tmp_path)
    log = tmp_path / "improvement_log.tsv"

    def _fake_eval(cmd):
        # the real CLI writes <out>.html + <out>.json; emulate the JSON twin
        out = Path(cmd[cmd.index("--out") + 1])
        payload = {
            "summaries": [
                {"bundle_dir": str(base), "reward_mean": 1.0},
                {"bundle_dir": str(treat), "reward_mean": 2.5},
            ],
            "eval_episodes_sha256": {str(base): "a" * 64, str(treat): "b" * 64},
        }
        out.with_suffix(".json").write_text(json.dumps(payload))
        out.write_text("<html></html>")
        return 0

    monkeypatch.setattr(eval_signed.subprocess, "call", _fake_eval)
    rc = eval_signed.main([
        "--baseline", str(base),
        "--treatment", str(treat),
        "--cartridge", "POKEMON CORAL",
        "--rom", "unused.gbc",
        "--episodes", "2",
        "--log", str(log),
    ])
    assert rc == 0

    rows = log.read_text().strip().splitlines()
    assert rows[0] == "ts\tbaseline\ttreatment\tscore\tepisodes_sha\tsig"
    parts = rows[1].split("\t")
    assert len(parts) == 6
    ts, b, t, score, episodes_sha, sig = parts
    assert float(score) == pytest.approx(2.5)  # the TREATMENT mean, not NaN
    # reviewer recomputation: digest must reproduce from the on-disk artifacts
    expected = eval_signed._digest(base, treat, episodes_sha, float(score), ts)
    assert sig == expected

    # ...and NOT reproduce after tampering with the score
    tampered = eval_signed._digest(base, treat, episodes_sha, 99.0, ts)
    assert tampered != sig


def test_missing_json_is_loud(tmp_path, monkeypatch):
    base, treat = _fake_bundles(tmp_path)
    monkeypatch.setattr(eval_signed.subprocess, "call", lambda cmd: 0)  # writes nothing
    rc = eval_signed.main([
        "--baseline", str(base),
        "--treatment", str(treat),
        "--cartridge", "X",
        "--rom", "x",
        "--log", str(tmp_path / "log.tsv"),
    ])
    assert rc == 3
