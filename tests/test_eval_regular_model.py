# ------------------------------------------------------------------------
# RF-DETR
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Training loop of Model.train with mocked evaluate/train_one_epoch (CPU, no data)."""
import json
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rfdetr import main as rf_main
from rfdetr.util.early_stopping import EarlyStoppingCallback

# Regular and EMA AP per epoch: the regular model beats the EMA in epoch 1, the EMA improves in epoch 2
REGULAR = [0.50, 0.70, 0.60]
EMA = [0.30, 0.40, 0.45]


def _stats(ap: float) -> dict:
    return {
        "coco_eval_bbox": [ap, ap + 0.1],
        "results_json": {"class_map": {"all": {"map@50:95": ap}}},
    }


def _run(monkeypatch, tmp_path, eval_regular_model: bool, resume: str = "", ema_seq=None, reg_seq=None):
    calls = {"regular": 0, "ema": 0}
    ema_holder = {}

    def fake_evaluate(model, criterion, postprocess, loader, base_ds, device, args=None, header="Test"):
        if model is ema_holder["m"].module:
            ap = (ema_seq or EMA)[calls["ema"]]
            calls["ema"] += 1
        else:
            ap = (reg_seq or REGULAR)[calls["regular"]]
            calls["regular"] += 1
        return _stats(ap), SimpleNamespace(coco_eval={})

    args = rf_main.populate_args(
        dataset_file="roboflow", num_classes=2, use_ema=True, epochs=3, output_dir=str(tmp_path), run_test=False,
        eval_regular_model=eval_regular_model, resume=resume, early_stopping=True, early_stopping_use_ema=True, device="cpu",
        num_workers=0, batch_size=1, grad_accum_steps=1, tensorboard=False, class_names=None, segmentation_head=False,
    )
    model = nn.Linear(2, 2)
    monkeypatch.setattr(rf_main, "populate_args", lambda **kw: args)
    monkeypatch.setattr(rf_main.utils, "init_distributed_mode", lambda a: setattr(a, "distributed", False))
    monkeypatch.setattr(rf_main, "build_criterion_and_postprocessors", lambda a: (nn.Module(), None))
    monkeypatch.setattr(rf_main, "build_dataset", lambda **kw: list(range(8)))
    monkeypatch.setattr(rf_main, "get_param_dict", lambda a, m: [{"params": p} for p in m.parameters()])
    monkeypatch.setattr(rf_main, "get_coco_api_from_dataset", lambda d: None)
    monkeypatch.setattr(rf_main, "train_one_epoch", lambda *a, **k: {"loss": 1.0})
    monkeypatch.setattr(rf_main, "evaluate", fake_evaluate)
    orig_ema = rf_main.ModelEma

    def make_ema(*a, **k):
        ema_holder["m"] = orig_ema(*a, **k)
        return ema_holder["m"]

    monkeypatch.setattr(rf_main, "ModelEma", make_ema)

    obj = object.__new__(rf_main.Model)
    obj.model = model
    obj.stop_early = False
    callbacks = defaultdict(list)
    es = EarlyStoppingCallback(model=obj, patience=10, min_delta=0.001, use_ema=True)
    callbacks["on_fit_epoch_end"].append(es.update)
    obj.train(callbacks=callbacks)
    return calls, tmp_path


def _log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "log.txt").read_text().splitlines()]


def test_skip_regular_eval_keeps_ema_eval_and_best(monkeypatch, tmp_path) -> None:
    calls, out = _run(monkeypatch, tmp_path, eval_regular_model=False)
    assert calls == {"regular": 0, "ema": 3}
    assert (out / "checkpoint.pth").exists() and (out / "checkpoint_best_ema.pth").exists()
    assert not (out / "checkpoint_best_regular.pth").exists()
    log = _log(out)
    assert len(log) == 3 and all("test_coco_eval_bbox" not in r for r in log)
    assert [r["ema_test_coco_eval_bbox"][0] for r in log] == EMA
    assert (out / "results.json").exists()
    assert torch.load(out / "checkpoint_best_ema.pth", weights_only=False)["epoch"] == 2


def test_default_evaluates_both_and_saves_ema_best_below_regular(monkeypatch, tmp_path) -> None:
    calls, out = _run(monkeypatch, tmp_path, eval_regular_model=True)
    assert calls == {"regular": 3, "ema": 3}
    assert (out / "checkpoint_best_regular.pth").exists()
    # The EMA improves its own maximum in epoch 2 even though the regular model had 0.70: it must be saved
    assert torch.load(out / "checkpoint_best_ema.pth", weights_only=False)["epoch"] == 2


def test_should_eval_regular_requires_ema() -> None:
    assert rf_main._should_eval_regular(SimpleNamespace(eval_regular_model=False, use_ema=False)) is True
    assert rf_main._should_eval_regular(SimpleNamespace(eval_regular_model=False, use_ema=True)) is False
    assert rf_main._should_eval_regular(SimpleNamespace(use_ema=True)) is True


def _prep_resume_dir(d):
    d.mkdir()
    model = nn.Linear(2, 2)
    opt = torch.optim.AdamW([{"params": p} for p in model.parameters()], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    ckpt = d / "checkpoint.pth"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "lr_scheduler": sched.state_dict(), "epoch": 1}, ckpt)
    # sentinel of the previous best (epoch -99)
    torch.save({"epoch": -99, "model": model.state_dict(), "args": SimpleNamespace()}, d / "checkpoint_best_ema.pth")
    rows = [{"epoch": e, "ema_test_coco_eval_bbox": [ap, 0.0]} for e, ap in enumerate([0.60, 0.65])]
    (d / "log.txt").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(ckpt)


def test_resume_does_not_overwrite_best_ema_with_worse_epoch(monkeypatch, tmp_path) -> None:
    """Al reanudar, el maximo EMA previo (log.txt) se conserva: un EMA peor no sobrescribe el best."""
    d = tmp_path / "peor"
    _run(monkeypatch, d, eval_regular_model=True, resume=_prep_resume_dir(d), ema_seq=[0.45], reg_seq=[0.01])
    assert torch.load(d / "checkpoint_best_ema.pth", weights_only=False)["epoch"] == -99


def test_resume_saves_best_ema_when_it_beats_previous_max(monkeypatch, tmp_path) -> None:
    d = tmp_path / "mejor"
    _run(monkeypatch, d, eval_regular_model=False, resume=_prep_resume_dir(d), ema_seq=[0.70])
    assert torch.load(d / "checkpoint_best_ema.pth", weights_only=False)["epoch"] == 2
