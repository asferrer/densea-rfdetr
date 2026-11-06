# Changes with respect to upstream

Upstream: [roboflow/rf-detr](https://github.com/roboflow/rf-detr) at commit
`750f1e7cf543d07e533f1c85def2d46e7a9f5b75` (version 1.5.0.rc2). The first commit of this repository is that tree,
without four files that are not part of the code base (the assistant-instruction files of upstream, its documentation
`CNAME` and its Copilot instructions). This file lists the files of
upstream that were modified (Apache-2.0, section 4b) and what was added or removed.

## Modified files

| File | Change |
|---|---|
| `src/rfdetr/config.py` | New training option `eval_regular_model` (default `True`). With `False` and `use_ema=True` the regular weights are not evaluated every epoch. |
| `src/rfdetr/main.py` | Honours `eval_regular_model`; pinned memory and persistent workers in the data loaders; on resume, restores the best metrics and the early-stopping counter from `log.txt`; GPU cache clean-up between epochs. |
| `src/rfdetr/engine.py` | GPU cache clean-up and garbage collection after each training epoch. |
| `src/rfdetr/util/utils.py` | `BestMetricHolder.update` returns whether the model (EMA or regular) improved its own maximum; new `restore_from_log`. |
| `src/rfdetr/util/early_stopping.py` | New `EarlyStoppingCallback.restore_from_log`: replays `log.txt` so that the patience counts the whole run. |
| `src/rfdetr/datasets/coco_eval.py` | Optional evaluation with faster-coco-eval, enabled with the environment variable `FAST_COCOEVAL=1`; falls back to pycocotools if the package is missing. |
| `src/rfdetr/models/lwdetr.py` | `update_drop_path` also handles the DINOv2 wrapper (`encoder.encoder.encoder.layer`). |
| `pyproject.toml` | Project URLs; the `docs` extra and the test-retry plugin are removed. |

## Added files

- `densea/hyperparams.py`, `densea/train.py`, `densea/prepare_dataset.py`, `densea/dump_test_predictions.py`:
  training setup, launcher, dataset converter and test-detections dump.
- `tests/test_eval_regular_model.py`, `tests/test_dump_ema.py`, `tests/util/test_best_metric_holder.py`,
  `tests/util/test_early_stopping_resume.py`: tests of the changes above (CPU only).
- `Dockerfile`, `docker-compose.yml`, `requirements-pinned.txt`, `.gitignore`, `.gitattributes`, `README.md`,
  `NOTICE`, `CHANGES.md`, `CITATION.cff`.

## Removed files

`docs/`, `mkdocs.yaml`, `.github/`, `.codecov.yml`, `.pre-commit-config.yaml`, `CITATION.cff` of upstream (replaced
by a new one), `tests/benchmarks/` (GPU convergence tests that download datasets).
