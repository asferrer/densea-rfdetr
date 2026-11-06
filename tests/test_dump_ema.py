# ------------------------------------------------------------------------
# RF-DETR
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""dump_test_predictions --ema: EMA weights of a periodic checkpoint."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "densea"))
import dump_test_predictions as dump  # noqa: E402


def test_without_ema_uses_model():
    ck = {"model": {"w": 1}, "ema_model": {"w": 2}}
    assert dump.model_state(ck, use_ema=False) == {"w": 1}


def test_with_ema_uses_ema_model():
    ck = {"model": {"w": 1}, "ema_model": {"w": 2}}
    assert dump.model_state(ck, use_ema=True) == {"w": 2}


def test_ema_without_ema_model_fails():
    with pytest.raises(SystemExit):
        dump.model_state({"model": {"w": 1}}, use_ema=True)
