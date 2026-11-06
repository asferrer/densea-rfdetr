# ------------------------------------------------------------------------
# RF-DETR
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests de BestMetricHolder: el best EMA y el best regular son independientes."""
from rfdetr.util.utils import BestMetricHolder


def test_ema_best_saved_when_improving_own_max_below_regular() -> None:
    """The regular model beats the EMA, then the EMA improves its own maximum without beating the regular model."""
    h = BestMetricHolder(use_ema=True)
    assert h.update(0.50, 0, is_ema=False) is True
    assert h.update(0.30, 0, is_ema=True) is True  # primer EMA: mejora su propio maximo
    assert h.update(0.55, 1, is_ema=False) is True
    assert h.update(0.40, 1, is_ema=True) is True  # 0.40 < 0.55 (regular) pero > 0.30 (EMA)
    assert h.update(0.35, 2, is_ema=True) is False  # no mejora su propio maximo
    assert h.best_ema.best_res == 0.40 and h.best_ema.best_ep == 1


def test_regular_best_independent_of_ema() -> None:
    h = BestMetricHolder(use_ema=True)
    assert h.update(0.60, 0, is_ema=True) is True
    assert h.update(0.40, 0, is_ema=False) is True  # regular mejora su propio maximo
    assert h.update(0.40, 1, is_ema=False) is False


def test_best_all_keeps_global_best_for_summary() -> None:
    h = BestMetricHolder(use_ema=True)
    h.update(0.55, 1, is_ema=False)
    h.update(0.40, 1, is_ema=True)
    assert h.best_all.best_res == 0.55 and h.best_all.best_ep == 1
    s = h.summary()
    assert s["ema_best_res"] == 0.40 and s["regular_best_res"] == 0.55 and s["all_best_res"] == 0.55


def test_no_ema_mode_unchanged() -> None:
    h = BestMetricHolder(use_ema=False)
    assert h.update(0.3, 0) is True
    assert h.update(0.2, 1) is False
    assert h.update(0.4, 2) is True


def test_restore_from_log_uses_only_epochs_before_resume(tmp_path) -> None:
    import json

    rows = [
        {"epoch": 0, "test_coco_eval_bbox": [0.5, 0], "ema_test_coco_eval_bbox": [0.3, 0]},
        {"epoch": 1, "test_coco_eval_bbox": [0.7, 0], "ema_test_coco_eval_bbox": [0.6, 0]},
        {"epoch": 2, "test_coco_eval_bbox": [0.9, 0], "ema_test_coco_eval_bbox": [0.9, 0]},  # >= before_epoch
    ]
    log = tmp_path / "log.txt"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"epoch": 3, "tru')  # linea truncada
    h = BestMetricHolder(use_ema=True)
    h.restore_from_log(log, before_epoch=2)
    assert (h.best_ema.best_res, h.best_regular.best_res, h.best_all.best_res) == (0.6, 0.7, 0.7)
    assert h.update(0.55, 2, is_ema=True) is False
    assert h.update(0.65, 2, is_ema=True) is True


def test_restore_from_log_missing_file(tmp_path) -> None:
    h = BestMetricHolder(use_ema=True)
    h.restore_from_log(tmp_path / "nope.txt", before_epoch=5)
    assert h.best_ema.best_res == 0.0
