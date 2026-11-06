# ------------------------------------------------------------------------
# RF-DETR
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""EarlyStoppingCallback keeps its best value and counter on resume (from log.txt)."""
import json
from types import SimpleNamespace

from rfdetr.util.early_stopping import EarlyStoppingCallback


def _row(epoch: int, ap: float) -> dict:
    # the regular model (0.99) is irrelevant: with use_ema=True only the EMA counts
    return {"epoch": epoch, "test_coco_eval_bbox": [0.99, 0.0], "ema_test_coco_eval_bbox": [ap, 0.0]}


def _write_log(path, aps) -> None:
    path.write_text("".join(json.dumps(_row(e, ap)) + "\n" for e, ap in enumerate(aps)))


class _Stopper:
    def __init__(self) -> None:
        self.stopped = False

    def request_early_stop(self) -> None:
        self.stopped = True


def _aps_best_at_10th_epoch():
    # 25 epocas, la mejor es la 10 (indice 9); el resto no mejora
    return [0.01 * (i + 1) for i in range(10)] + [0.05] * 15


def test_restore_counter_and_stop_after_five_more_epochs(tmp_path) -> None:
    log = tmp_path / "log.txt"
    _write_log(log, _aps_best_at_10th_epoch())
    stopper = _Stopper()
    cb = EarlyStoppingCallback(model=stopper, patience=20, min_delta=0.001, use_ema=True)
    cb.restore_from_log(log, before_epoch=25)  # resume at epoch 26 (index 25)
    assert cb.counter == 15 and abs(cb.best_map - 0.10) < 1e-9
    for e in range(25, 29):
        cb.update(_row(e, 0.05))
        assert not stopper.stopped
    cb.update(_row(29, 0.05))
    assert cb.counter == 20 and stopper.stopped


def test_restore_equals_uninterrupted_run(tmp_path) -> None:
    aps = _aps_best_at_10th_epoch()
    full = EarlyStoppingCallback(model=None, patience=20, min_delta=0.001, use_ema=True, verbose=False)
    for e, ap in enumerate(aps):
        full.update(_row(e, ap))
    log = tmp_path / "log.txt"
    _write_log(log, aps)
    resumed = EarlyStoppingCallback(model=None, patience=20, min_delta=0.001, use_ema=True, verbose=False)
    resumed.restore_from_log(log, before_epoch=25)
    assert (resumed.best_map, resumed.counter) == (full.best_map, full.counter)


def test_restore_ignores_epochs_after_resume_and_truncated_line(tmp_path) -> None:
    log = tmp_path / "log.txt"
    _write_log(log, [0.1, 0.2, 0.3, 0.9])
    with log.open("a") as f:
        f.write('{"epoch": 4, "ema_te')
    cb = EarlyStoppingCallback(model=None, patience=20, use_ema=True, verbose=False)
    cb.restore_from_log(log, before_epoch=3)
    assert cb.best_map == 0.3 and cb.counter == 0
    cb.restore_from_log(tmp_path / "nope.txt", before_epoch=3)  # no log: does not fail
