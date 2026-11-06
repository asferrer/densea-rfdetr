# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Early stopping callback for RF-DETR training
"""

import json
import os
from typing import Any, Dict

from rfdetr.util.logger import get_logger

logger = get_logger()


class EarlyStoppingCallback:
    """
    Early stopping callback that monitors mAP and stops training if no improvement
    over a threshold is observed for a specified number of epochs.

    Args:
        patience (int): Number of epochs with no improvement to wait before stopping
        min_delta (float): Minimum change in mAP to qualify as improvement
        use_ema (bool): Whether to use EMA model metrics for early stopping
        verbose (bool): Whether to print early stopping messages
    """

    def __init__(
        self,
        model: Any,
        patience: int = 5,
        min_delta: float = 0.001,
        use_ema: bool = False,
        verbose: bool = True,
        segmentation_head: bool = False,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.use_ema = use_ema
        self.verbose = verbose
        self.best_map = 0.0
        self.counter = 0
        self.model = model
        self.segmentation_head = segmentation_head

    def restore_from_log(self, log_path, before_epoch: int) -> None:
        """Restore ``best_map`` and ``counter`` when resuming, replaying ``update`` over ``log.txt``.

        Only epochs < ``before_epoch`` are replayed, with the same metric and ``min_delta`` logic as
        ``update``, so the patience counts the whole run and not just the epochs since the resume.
        """
        if not os.path.exists(log_path):
            return
        model, verbose = self.model, self.verbose
        self.model, self.verbose = None, False  # replay: no early-stop request, no per-epoch logging
        self.best_map, self.counter = 0.0, 0
        try:
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # truncated last line after a crash
                    if row.get("epoch") is None or row["epoch"] >= before_epoch:
                        continue
                    if not any(k in row for k in ("test_coco_eval_bbox", "ema_test_coco_eval_bbox", "test_coco_eval_masks", "ema_test_coco_eval_masks")):
                        continue
                    self.update(row)
        finally:
            self.model, self.verbose = model, verbose
        logger.info(f"Early stopping restored from log: best mAP {self.best_map:.4f}, counter {self.counter}")

    def update(self, log_stats: Dict[str, Any]) -> None:
        """Update early stopping state based on epoch validation metrics"""
        regular_map = None
        ema_map = None

        if "test_coco_eval_bbox" in log_stats:
            if not self.segmentation_head:
                regular_map = log_stats["test_coco_eval_bbox"][0]
            else:
                regular_map = log_stats["test_coco_eval_masks"][0]

        if "ema_test_coco_eval_bbox" in log_stats:
            if not self.segmentation_head:
                ema_map = log_stats["ema_test_coco_eval_bbox"][0]
            else:
                ema_map = log_stats["ema_test_coco_eval_masks"][0]

        current_map = None
        if regular_map is not None and ema_map is not None:
            if self.use_ema:
                current_map = ema_map
                metric_source = "EMA"
            else:
                current_map = max(regular_map, ema_map)
                metric_source = "max(regular, EMA)"
        elif ema_map is not None:
            current_map = ema_map
            metric_source = "EMA"
        elif regular_map is not None:
            current_map = regular_map
            metric_source = "regular"
        else:
            if self.verbose:
                raise ValueError("No valid mAP metric found!")
            return

        if self.verbose:
            logger.info(
                f"Early stopping: Current mAP ({metric_source}): {current_map:.4f}, Best: {self.best_map:.4f}, Diff: {current_map - self.best_map:.4f}, Min delta: {self.min_delta}"
            )
        if current_map > self.best_map + self.min_delta:
            self.best_map = current_map
            self.counter = 0
            logger.info(f"Early stopping: mAP improved to {current_map:.4f} using {metric_source} metric")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"Early stopping: No improvement in mAP for {self.counter} epochs (best: {self.best_map:.4f}, current: {current_map:.4f})"
                )
        if self.counter >= self.patience:
            logger.info(
                f"Early stopping triggered: No improvement above {self.min_delta} threshold for {self.patience} epochs"
            )
            if self.model:
                self.model.request_early_stop()
