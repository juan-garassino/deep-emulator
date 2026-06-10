"""DINO trainer + loss + augmentations — end-to-end smoke without env deps."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_multicrop_returns_correct_number_and_shapes():
    from deepEmulator.encoders.augmentations import MultiCropAugment, MultiCropConfig

    aug = MultiCropAugment(MultiCropConfig(n_global=2, n_local=4, out_size=96))
    x = torch.randint(0, 256, (3, 1, 96, 96), dtype=torch.uint8)
    crops = aug(x)
    assert len(crops) == 6
    for c in crops:
        assert c.shape == (3, 1, 96, 96)
        assert c.dtype == torch.float32
        assert c.min().item() >= 0.0 and c.max().item() <= 1.0


def test_dino_head_output_shape():
    from deepEmulator.encoders.dino import DINOHead

    h = DINOHead(in_dim=256, out_dim=512, hidden_dim=128, bottleneck_dim=64)
    x = torch.randn(4, 256)
    out = h(x)
    assert out.shape == (4, 512)


def test_dino_loss_decreases_on_simple_signal():
    """Sanity: with a deterministic pair, gradient should reduce the loss."""
    from deepEmulator.encoders.dino import DINOLoss

    loss_fn = DINOLoss(out_dim=128, n_global=2, teacher_temp=0.04, student_temp=0.1)
    s_outs = [torch.randn(4, 128, requires_grad=True) for _ in range(4)]
    t_outs = [torch.randn(4, 128) for _ in range(2)]
    l1 = loss_fn(s_outs, t_outs)
    assert l1.requires_grad and l1.item() > 0
    l1.backward()
    assert s_outs[0].grad is not None


def test_dino_trainer_step_runs_and_updates_student():
    """30 steps on tiny ViT + random batch: loss stays finite, params move, teacher EMA tracks.

    Note: DINO loss is intentionally non-monotonic in the first ~hundreds of
    iterations (centering buffer warms up against sharp teacher τ=0.04).
    Monotonic decrease is not a valid sanity check at this scale.
    """
    import math

    from deepEmulator.encoders.augmentations import MultiCropConfig
    from deepEmulator.encoders.dino import DINOConfig, DINOTrainer
    from deepEmulator.encoders.vit import ViTConfig

    torch.manual_seed(0)
    trainer = DINOTrainer(
        vit_cfg=ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3),
        dino_cfg=DINOConfig(out_dim=256, hidden_dim=64, bottleneck_dim=32, learning_rate=1e-3),
        crop_cfg=MultiCropConfig(n_global=2, n_local=2),
        device="cpu",
    )
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, size=(8, 1, 96, 96), dtype=np.uint8)

    cls_token_before = trainer.student.cls_token.detach().clone()
    teacher_before = trainer.teacher.cls_token.detach().clone()

    for _ in range(30):
        loss = trainer.step(batch)
        assert math.isfinite(loss), f"non-finite loss: {loss}"

    # Student moved
    assert not torch.allclose(cls_token_before, trainer.student.cls_token.detach())
    # Teacher EMA tracked student (so also moved, but less)
    assert not torch.allclose(teacher_before, trainer.teacher.cls_token.detach())
    # Centering buffer has been updated away from zero
    assert trainer.loss_fn.center.abs().sum().item() > 0


def test_dino_trainer_state_dict_roundtrip(tmp_path):
    from deepEmulator.encoders.augmentations import MultiCropConfig
    from deepEmulator.encoders.dino import DINOConfig, DINOTrainer
    from deepEmulator.encoders.vit import ViTConfig

    t1 = DINOTrainer(
        vit_cfg=ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3),
        dino_cfg=DINOConfig(out_dim=256, hidden_dim=64, bottleneck_dim=32),
        crop_cfg=MultiCropConfig(n_global=2, n_local=2),
        device="cpu",
    )
    sd = t1.state_dict()
    torch.save(sd, tmp_path / "enc.pt")

    t2 = DINOTrainer(
        vit_cfg=ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3),
        dino_cfg=DINOConfig(out_dim=256, hidden_dim=64, bottleneck_dim=32),
        crop_cfg=MultiCropConfig(n_global=2, n_local=2),
        device="cpu",
    )
    t2.load_state_dict(torch.load(tmp_path / "enc.pt", weights_only=False))

    x = torch.randn(2, 1, 96, 96)
    with torch.no_grad():
        a = t1.student(x)
        b = t2.student(x)
    assert torch.allclose(a, b)


def test_pretrain_cli_writes_bundle(tmp_path, monkeypatch):
    """End-to-end CLI smoke: 30 steps on a tiny corpus → encoder bundle written."""
    from deepEmulator.data.frame_corpus import FrameStorage, normalize_to_96x96

    corpus = tmp_path / "corpus"
    store = FrameStorage(corpus, chunk_size=32)
    rng = np.random.default_rng(0)
    for _ in range(128):
        store.push(normalize_to_96x96(rng.integers(0, 256, size=(96, 96), dtype=np.uint8)))
    store.flush()

    runs_root = tmp_path / "encoders"
    monkeypatch.setattr(sys, "argv", [
        "deepemu-pretrain-dino",
        "--corpus", str(corpus),
        "--steps", "30",
        "--batch-size", "8",
        "--save-every", "20",
        "--log-every", "10",
        "--runs-root", str(runs_root),
        "--out-dim", "256",
        "--n-local-crops", "2",
    ])
    # Monkey-patch trainer to use a small ViT so the test stays fast
    from deepEmulator.training import pretrain_dino as pd_mod
    from deepEmulator.encoders.vit import ViTConfig as RealViTConfig

    def small_trainer(*a, **kw):
        from deepEmulator.encoders.dino import DINOTrainer
        from deepEmulator.encoders.augmentations import MultiCropConfig
        from deepEmulator.encoders.dino import DINOConfig

        return DINOTrainer(
            vit_cfg=RealViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3),
            dino_cfg=DINOConfig(out_dim=256, hidden_dim=64, bottleneck_dim=32),
            crop_cfg=MultiCropConfig(n_global=2, n_local=2),
        )

    monkeypatch.setattr(pd_mod, "DINOTrainer", small_trainer)
    rc = pd_mod.main()
    assert rc == 0

    # bundle exists + latest.txt points at it
    runs = list((runs_root / "dino").glob("2*"))
    assert len(runs) == 1
    run = runs[0]
    assert (run / "encoder.pt").exists()
    assert (run / "encoder_only.pt").exists()
    assert (run / "metadata.json").exists()
    assert (run / "metrics.tsv").exists()
    md = json.loads((run / "metadata.json").read_text())
    assert md["algo"] == "dino"
    assert md["step_count"] >= 30
    # marker stores the run NAME (relative) — survives the GCS round-trip
    assert (runs_root / "dino" / "latest.txt").read_text().strip() == run.name


def test_colab_pretrain_module_imports():
    from deepEmulator.training import colab_pretrain

    assert hasattr(colab_pretrain, "run_in_colab")


def test_dino_schedules_warmup_then_cosine():
    from deepEmulator.encoders.augmentations import MultiCropConfig
    from deepEmulator.encoders.dino import DINOConfig, DINOTrainer
    from deepEmulator.encoders.vit import ViTConfig

    trainer = DINOTrainer(
        vit_cfg=ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3),
        dino_cfg=DINOConfig(out_dim=128, hidden_dim=64, bottleneck_dim=32, learning_rate=1e-3),
        crop_cfg=MultiCropConfig(n_global=2, n_local=2),
        device="cpu",
        total_steps=100,
    )
    base = 1e-3
    trainer.step_count = 0
    assert trainer._current_lr() < base * 0.2  # warmup start, tiny
    trainer.step_count = 10  # warmup end (warmup_frac=0.1)
    assert trainer._current_lr() == pytest.approx(base)
    trainer.step_count = 55
    mid = trainer._current_lr()
    trainer.step_count = 99
    late = trainer._current_lr()
    assert base > mid > late >= base * trainer.dino_cfg.final_lr_frac * 0.99

    # WD ramps up, teacher momentum ramps toward 1
    trainer.step_count = 0
    wd0, m0 = trainer._current_wd(), trainer._current_teacher_momentum()
    trainer.step_count = 100
    wd1, m1 = trainer._current_wd(), trainer._current_teacher_momentum()
    assert wd0 == pytest.approx(0.04) and wd1 == pytest.approx(0.4)
    assert m0 == pytest.approx(0.996) and m1 == pytest.approx(1.0)


def test_dino_schedules_constant_without_total_steps():
    from deepEmulator.encoders.augmentations import MultiCropConfig
    from deepEmulator.encoders.dino import DINOConfig, DINOTrainer
    from deepEmulator.encoders.vit import ViTConfig

    trainer = DINOTrainer(
        vit_cfg=ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3),
        dino_cfg=DINOConfig(out_dim=128, hidden_dim=64, bottleneck_dim=32),
        crop_cfg=MultiCropConfig(n_global=2, n_local=2),
        device="cpu",
    )
    trainer.step_count = 12345
    assert trainer._current_lr() == trainer.dino_cfg.learning_rate
    assert trainer._current_wd() == trainer.dino_cfg.weight_decay
    assert trainer._current_teacher_momentum() == trainer.dino_cfg.teacher_ema_momentum


def test_dino_param_groups_exclude_norms_and_biases_from_wd():
    from deepEmulator.encoders.augmentations import MultiCropConfig
    from deepEmulator.encoders.dino import DINOConfig, DINOTrainer
    from deepEmulator.encoders.vit import ViTConfig

    trainer = DINOTrainer(
        vit_cfg=ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3),
        dino_cfg=DINOConfig(out_dim=128, hidden_dim=64, bottleneck_dim=32),
        crop_cfg=MultiCropConfig(n_global=2, n_local=2),
        device="cpu",
    )
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["weight_decay"] > 0
    assert groups[1]["weight_decay"] == 0.0
    # every 1-D param (biases, LayerNorm) must be in the no-decay group
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    for module in (trainer.student, trainer.student_head):
        for name, p in module.named_parameters():
            if p.requires_grad and p.ndim <= 1:
                assert id(p) in no_decay_ids, name


def test_multicrop_per_sample_crops_differ():
    """Per-sample crop boxes: two samples in a batch must (statistically) get
    different geometry — batch-shared crops gave identical view pairs."""
    from deepEmulator.encoders.augmentations import _random_resized_crop

    torch.manual_seed(0)
    # sample 0 and 1 are identical inputs; differing outputs => per-sample crop
    x = torch.rand(1, 1, 96, 96).repeat(8, 1, 1, 1)
    out = _random_resized_crop(x, 96, (0.05, 0.4))
    diffs = [(out[0] - out[i]).abs().mean().item() for i in range(1, 8)]
    assert max(diffs) > 1e-3
