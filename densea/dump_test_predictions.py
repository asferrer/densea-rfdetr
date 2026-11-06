#!/usr/bin/env python3
"""
Re-run RF-DETR inference on an experiment's TEST split and dump the raw
per-image detections in COCO format, so downstream analyses (bootstrap
confidence intervals, per-class error analysis) no longer need a re-run.

The model is rebuilt from the `args` object stored inside the checkpoint, so
the architecture, resolution and dataset paths are exactly those used at
training time. The evaluation loop is `rfdetr.engine.evaluate`, unchanged; the
only addition is a hook on CocoEvaluator.prepare that accumulates the COCO
detection dicts it produces batch by batch.

Usage (from the repository root):
    python densea/dump_test_predictions.py --checkpoint runs/real_only/checkpoint_best_ema.pth
        --output-dir runs/real_only/eval_test --dataset-dir /runs/rfdetr_data/real_only
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, SequentialSampler

from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util import misc as utils

COLLECTED: list = []


def model_state(checkpoint: dict, use_ema: bool) -> dict:
    """Weights to evaluate. Periodic checkpoints (checkpointNNNN.pth, checkpoint.pth) keep the
    regular model in "model" and the EMA in "ema_model"; checkpoint_best_ema.pth already stores
    the EMA in "model"."""
    if not use_ema:
        return checkpoint["model"]
    if "ema_model" not in checkpoint:
        raise SystemExit("--ema: the checkpoint has no 'ema_model' (checkpoint_best_ema.pth needs no --ema)")
    return checkpoint["ema_model"]


def patch_coco_evaluator() -> None:
    """Accumulate every COCO detection dict CocoEvaluator builds per batch."""
    original_prepare = CocoEvaluator.prepare

    def prepare(self, predictions, iou_type):
        results = original_prepare(self, predictions, iou_type)
        if iou_type == "bbox" and results:
            COLLECTED.extend(results)
        return results

    CocoEvaluator.prepare = prepare


def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", default=None, help="Override the dataset dir stored in the checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ema", action="store_true", help="evaluate the EMA weights of a periodic checkpoint")
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu, e.g. while the GPU is training")
    args_cli = parser.parse_args()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    checkpoint["model"] = model_state(checkpoint, args_cli.ema)

    args.pretrain_weights = None
    args.device = args_cli.device
    args.distributed = False
    args.world_size = 1
    args.batch_size = args_cli.batch_size
    args.num_workers = args_cli.num_workers
    args.output_dir = str(out_dir)
    if args_cli.dataset_dir:
        args.dataset_dir = args_cli.dataset_dir
    for name, default in (
        ("fp16_eval", False),
        ("segmentation_head", False),
        ("eval_max_dets", 100),
        ("print_freq", 50),
        ("progress_bar", False),
        ("amp", False),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    if args.device == "cpu":
        args.amp = False  # autocast is hard-wired to device_type="cuda"; evaluate in fp32
        args.fp16_eval = False

    device = torch.device(args.device)

    model = build_model(args)

    # Mirror rfdetr.main.Model.__init__: the detection head is sized from the
    # checkpoint, not from args.num_classes, because training reinitialised it.
    checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    if checkpoint_num_classes != args.num_classes + 1:
        model.reinitialize_detection_head(checkpoint_num_classes)

    num_desired_queries = args.num_queries * args.group_detr
    for name in ("refpoint_embed.weight", "query_feat.weight"):
        for key, state in list(checkpoint["model"].items()):
            if key.endswith(name):
                checkpoint["model"][key] = state[:num_desired_queries]

    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    criterion, postprocess = build_criterion_and_postprocessors(args)
    criterion.to(device)

    dataset = build_dataset(image_set=args_cli.split, args=args, resolution=args.resolution)
    base_ds = get_coco_api_from_dataset(dataset)
    loader = DataLoader(
        dataset,
        args.batch_size,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Checkpoint:  {args_cli.checkpoint}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Split:       {args_cli.split}  ({len(dataset)} images)")
    print(f"Resolution:  {args.resolution}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    patch_coco_evaluator()
    stats, _ = evaluate(model, criterion, postprocess, loader, base_ds, device, args=args, header="Test")

    # 300 detections per image over 2185 images is ~650k entries; full float
    # repr would be ~250 MB per experiment, so round to a precision that is
    # still far below one pixel / one score quantum.
    compact = [
        {
            "image_id": int(d["image_id"]),
            "category_id": int(d["category_id"]),
            "bbox": [round(float(v), 2) for v in d["bbox"]],
            "score": round(float(d["score"]), 6),
        }
        for d in COLLECTED
    ]
    pred_path = out_dir / "coco_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(compact, f, separators=(",", ":"))

    coco_bbox = stats.get("coco_eval_bbox")
    summary = {
        "checkpoint": args_cli.checkpoint,
        "dataset_dir": args.dataset_dir,
        "split": args_cli.split,
        "resolution": args.resolution,
        "gt_images": len(base_ds.dataset["images"]),
        "gt_annotations": len(base_ds.dataset["annotations"]),
        "num_predictions": len(COLLECTED),
        "coco_eval_bbox": coco_bbox,
        "AP": coco_bbox[0] if coco_bbox else None,
        "AP50": coco_bbox[1] if coco_bbox else None,
        "AP75": coco_bbox[2] if coco_bbox else None,
        "results_json": to_native(stats.get("results_json")),
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"GT images:       {summary['gt_images']}")
    print(f"GT annotations:  {summary['gt_annotations']}")
    print(f"Predictions:     {summary['num_predictions']}")
    if coco_bbox:
        print(f"AP:   {coco_bbox[0]:.4f}")
        print(f"AP50: {coco_bbox[1]:.4f}")
        print(f"AP75: {coco_bbox[2]:.4f}")
    print(f"Saved: {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
