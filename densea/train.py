"""Train RF-DETR (Base) on a COCO-format dataset and evaluate the best EMA model on the test split.

The dataset directory must contain ``train/``, ``valid/`` and ``test/``, each with an
``_annotations.coco.json`` (file names may be absolute paths or relative to the split directory).

    python densea/train.py --dataset-dir /runs/rfdetr_data/real_only --output-dir runs/real_only --seed 21

Training resumes from ``<output-dir>/checkpoint.pth`` if it exists. After training, the best EMA
checkpoint is evaluated on the test split with ``densea/dump_test_predictions.py``; the detections
are written to ``<output-dir>/eval_test/coco_predictions.json``.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Same results as pycocotools (difference <= 2e-16) but faster; set FAST_COCOEVAL=0 to disable.
os.environ.setdefault("FAST_COCOEVAL", "1")

from densea.hyperparams import get_hyperparams  # noqa: E402


def train(dataset_dir: str, output_dir: str, seed: int, run_name: str) -> None:
    import rfdetr

    hp = get_hyperparams(dataset_dir, output_dir, seed, run_name)
    if os.path.exists(f"{output_dir}/checkpoint.pth"):
        hp["resume"] = f"{output_dir}/checkpoint.pth"
    print({k: v for k, v in hp.items() if k != "aug_config"}, flush=True)
    rfdetr.RFDETRBase().train(**hp)


def evaluate_test(dataset_dir: str, output_dir: str) -> int:
    cmd = [sys.executable, str(ROOT / "densea" / "dump_test_predictions.py"),
           "--checkpoint", f"{output_dir}/checkpoint_best_ema.pth",
           "--output-dir", f"{output_dir}/eval_test", "--dataset-dir", dataset_dir]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--name", default=None, help="run name (default: output directory name)")
    ap.add_argument("--eval-only", action="store_true", help="skip training, evaluate the existing checkpoint")
    ap.add_argument("--no-test", action="store_true", help="do not evaluate on the test split")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.eval_only:
        train(args.dataset_dir, args.output_dir, args.seed, args.name or Path(args.output_dir).name)
    if not args.no_test:
        sys.exit(evaluate_test(args.dataset_dir, args.output_dir))


if __name__ == "__main__":
    main()
