"""Training setup used for the RF-DETR (Base) results on DenSea-Aggregated.

``get_hyperparams`` returns the keyword arguments passed to ``RFDETRBase().train()``;
``AUG_UNDERWATER`` is the albumentations preset used in every run.
"""
from typing import Any, Dict

# Keyword arguments of rfdetr's TrainConfig (extra fields are rejected by pydantic).
BASE_HYPERPARAMS: Dict[str, Any] = {
    # Optimizer
    "lr": 0.00005,
    "lr_encoder": 0.000005,
    "weight_decay": 0.05,
    # Schedule: 200 epochs, 6 warm-up epochs, cosine decay
    "epochs": 200,
    "warmup_epochs": 6.0,
    "lr_drop": 100,
    "lr_scheduler": "cosine",
    # Effective batch size 32 = 8 x 4 gradient accumulation steps
    "batch_size": 8,
    "grad_accum_steps": 4,
    "num_workers": 4,
    # EMA
    "use_ema": True,
    "ema_decay": 0.9998,
    # Early stopping on the validation AP of the EMA weights
    "early_stopping": True,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.001,
    # Multi-scale training
    "multi_scale": True,
    "expanded_scales": True,
    # Logging / checkpointing
    "tensorboard": True,
    "checkpoint_interval": 10,
}

# Albumentations preset for underwater imagery (keys are transform names).
AUG_UNDERWATER: Dict[str, Any] = {
    "HorizontalFlip": {"p": 0.5},
    "Rotate": {"limit": 15, "p": 0.4},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.3, "hue": 0.15, "p": 0.5},
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},
    "GaussNoise": {"std_range": (0.01, 0.03), "p": 0.3},
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.4},
}


def get_hyperparams(dataset_dir: str, output_dir: str, seed: int = 21, run_name: str = "run") -> Dict[str, Any]:
    """Full set of ``train()`` arguments for one run.

    Model selection uses the EMA weights only: ``eval_regular_model=False`` skips the per-epoch
    evaluation of the regular weights, ``early_stopping_use_ema=True`` makes the patience follow the
    EMA AP and ``checkpoint_best_ema.pth`` is the model that is evaluated on the test split.
    ``run_test=False`` keeps the test split untouched during training.
    """
    hp = dict(BASE_HYPERPARAMS)
    hp.update({
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "aug_config": AUG_UNDERWATER,
        "project": "DenSea-Aggregated",
        "run": run_name,
        "device": "cuda",
        "seed": seed,
        "early_stopping_use_ema": True,
        "eval_regular_model": False,
        "run_test": False,
    })
    return hp
