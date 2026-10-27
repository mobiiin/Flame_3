import argparse
import copy
import gc
import json
import os
import time
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import optuna

from data_loading import DATASET_PATHS, UnifiedFlameDataset, get_default_transform

# ==============================================================================
# CONFIGURATION & PLOT STYLING
# ==============================================================================
SEED = 42
NUM_WORKERS = 4
USE_SUPCON = True
TRAIN_DATA_FRACTION = 1.0
EVAL_MODE = True
FUSION_MODE = "EARLY"  # Options: "EARLY", "LATE"
COMMON_TEST_DATASETS = ["TEST_SET"]
USE_OPTUNA_CV = False
DEFAULT_HYPERPARAMS = {
    "base_filters": 16,
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "dropout": 0.2,
    "supcon_temperature": 0.07,
    "supcon_lambda": 0.1,
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    datasets: List[str]
    modality: str
    thermal_source: Optional[str] = None
    image_size: int = 224
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    optuna_trials: int = 10
    cv_epochs: int = 20
    final_epochs: int = 40
    batch_size_candidates: Tuple[int, ...] = (16, 32, 64)
    auto_batch_size: bool = False
    target_gpu_mem_util: float = 0.9
    max_auto_batch_size: int = 512
    quick_sweep: bool = False


EXPERIMENT_PRESETS = {
    "baseline_rgb_all": ExperimentConfig(
        name="baseline_rgb_all",
        datasets=["FLAME1", "FLAME2", "FLAME3"],
        modality="RGB",
    ),
    "flame3_rgb": ExperimentConfig(
        name="flame3_rgb",
        datasets=["FLAME3"],
        modality="RGB",
    ),
    "flame3_rgbt_jpg": ExperimentConfig(
        name="flame3_rgbt_jpg",
        datasets=["FLAME3"],
        modality="RGBT",
        thermal_source="jpg",
    ),
    "flame3_rgbt_tiff": ExperimentConfig(
        name="flame3_rgbt_tiff",
        datasets=["FLAME3"],
        modality="RGBT",
        thermal_source="tiff",
    ),
    "flame3_thermal_tiff": ExperimentConfig(
        name="flame3_thermal_tiff",
        datasets=["FLAME3"],
        modality="THERMAL",
        thermal_source="tiff",
    ),
}

# Publication-ready plot styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run BFLAME dataset ablation experiments.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_PRESETS.keys()), default="flame3_rgbt_tiff")
    parser.add_argument("--run-name")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_PATHS.keys()))
    parser.add_argument("--modality", choices=["RGB", "THERMAL", "TIFF", "RGBT"])
    parser.add_argument("--thermal-source", choices=["jpg", "tiff"])
    parser.add_argument("--fusion-mode", choices=["EARLY", "LATE", "early", "late"])
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--optuna-trials", type=int)
    parser.add_argument("--cv-epochs", type=int)
    parser.add_argument("--final-epochs", type=int)
    parser.add_argument("--batch-size-candidates", nargs="+", type=int, help="Batch sizes for Optuna search (e.g., 32 64 96 128).")
    parser.add_argument("--quick-sweep", action="store_true", help="Reduce hyperparameter sweep cost for faster turnaround.")
    parser.add_argument("--auto-batch-size", action="store_true", help="Auto-tune batch size to target GPU VRAM utilization.")
    parser.add_argument("--target-gpu-mem-util", type=float, default=0.9, help="Target GPU VRAM utilization for auto batch-size (0-1).")
    parser.add_argument("--max-auto-batch-size", type=int, default=512, help="Max batch size to probe in auto batch-size mode.")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--outdir", default="eval_results", help="Output directory for publication-ready results.")
    parser.add_argument("--dry-run", action="store_true", help="Run on a small subset to verify pipeline works end-to-end.")
    parser.add_argument("--dry-run-fraction", type=float, default=0.1, help="Fraction of data to use in dry-run mode (default: 0.1 = 10 percent).")
    parser.add_argument(
        "--use-optuna-cv",
        action="store_true",
        help="Enable Optuna hyperparameter sweep before final training.",
    )
    parser.add_argument(
        "--no-optuna-cv",
        action="store_true",
        help="Disable Optuna sweep and use DEFAULT_HYPERPARAMS directly.",
    )
    parser.add_argument("--list-experiments", action="store_true")
    return parser.parse_args()


def resolve_experiment_config(args):
    if args.use_optuna_cv and args.no_optuna_cv:
        raise ValueError("Cannot pass both --use-optuna-cv and --no-optuna-cv.")

    config = EXPERIMENT_PRESETS[args.experiment]
    name_overridden = False

    if args.datasets:
        config = replace(config, datasets=args.datasets)
        name_overridden = True
    if args.modality:
        config = replace(config, modality=args.modality)
        name_overridden = True
    if args.thermal_source:
        config = replace(config, thermal_source=args.thermal_source)
        name_overridden = True
    if args.image_size:
        config = replace(config, image_size=args.image_size)
    if args.val_ratio is not None:
        config = replace(config, val_ratio=args.val_ratio)
    if args.test_ratio is not None:
        config = replace(config, test_ratio=args.test_ratio)
    if args.optuna_trials:
        config = replace(config, optuna_trials=args.optuna_trials)
    if args.cv_epochs:
        config = replace(config, cv_epochs=args.cv_epochs)
    if args.final_epochs:
        config = replace(config, final_epochs=args.final_epochs)
    if args.batch_size_candidates:
        clean_candidates = sorted({int(x) for x in args.batch_size_candidates if int(x) > 0})
        if not clean_candidates:
            raise ValueError("batch_size_candidates must contain at least one positive integer.")
        config = replace(config, batch_size_candidates=tuple(clean_candidates))
    if args.quick_sweep:
        config = replace(config, quick_sweep=True)
    if args.auto_batch_size:
        config = replace(config, auto_batch_size=True)
    if args.target_gpu_mem_util is not None:
        config = replace(config, target_gpu_mem_util=args.target_gpu_mem_util)
    if args.max_auto_batch_size is not None:
        config = replace(config, max_auto_batch_size=args.max_auto_batch_size)

    if config.modality == "RGB" and config.thermal_source is not None:
        config = replace(config, thermal_source=None)

    if args.run_name:
        config = replace(config, name=args.run_name)
    elif name_overridden:
        dataset_part = "_".join(dataset.lower() for dataset in config.datasets)
        modality_part = config.modality.lower()
        if config.thermal_source:
            modality_part = f"{modality_part}_{config.thermal_source}"
        config = replace(config, name=f"{dataset_part}__{modality_part}")

    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if not 0.0 < config.test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")
    if config.val_ratio + config.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")
    if not 0.1 <= config.target_gpu_mem_util <= 0.98:
        raise ValueError("target_gpu_mem_util must be between 0.1 and 0.98.")
    if config.max_auto_batch_size < 8:
        raise ValueError("max_auto_batch_size must be >= 8.")

    if config.quick_sweep:
        reduced_trials = min(config.optuna_trials, 8)
        reduced_cv_epochs = min(config.cv_epochs, 8)
        reduced_batch_candidates = tuple(sorted(set(config.batch_size_candidates[:2] or (16, 32))))
        config = replace(
            config,
            optuna_trials=reduced_trials,
            cv_epochs=reduced_cv_epochs,
            batch_size_candidates=reduced_batch_candidates,
        )

    return config


def print_experiment_catalog():
    print("Available experiment presets:")
    for name, config in EXPERIMENT_PRESETS.items():
        thermal = config.thermal_source or "n/a"
        print(
            f"  {name}: datasets={config.datasets}, modality={config.modality}, "
            f"thermal_source={thermal}, val_ratio={config.val_ratio}, test_ratio={config.test_ratio}"
        )


def print_run_config(config):
    thermal = config.thermal_source or "n/a"
    print("\n--- Experiment Configuration ---")
    print(f"Name: {config.name}")
    print(f"Datasets: {config.datasets}")
    print(f"Modality: {config.modality}")
    print(f"Thermal source: {thermal}")
    print(f"Image size: {config.image_size}")
    print(f"Validation ratio: {config.val_ratio}")
    print(f"Test ratio: {config.test_ratio}")
    print(f"Optuna trials: {config.optuna_trials}")
    print(f"CV epochs: {config.cv_epochs}")
    print(f"Final epochs: {config.final_epochs}")
    print(f"Batch-size candidates: {list(config.batch_size_candidates)}")
    print(f"Quick sweep: {config.quick_sweep}")
    print(f"Auto batch-size: {config.auto_batch_size}")
    if config.auto_batch_size:
        print(f"Target GPU VRAM utilization: {config.target_gpu_mem_util:.2f}")
        print(f"Max auto batch size: {config.max_auto_batch_size}")


def stratified_train_val_indices(labels, val_ratio, seed):
    all_indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return train_idx, val_idx


def validate_split_feasibility(labels, val_ratio):
    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < 2:
        raise RuntimeError("Need at least 2 samples per class to create train and validation splits.")

    train_idx, val_idx = stratified_train_val_indices(labels, val_ratio, SEED)
    split_sizes = {
        "train": len(train_idx),
        "val": len(val_idx),
    }

    for split_name, indices in (("train", train_idx), ("val", val_idx)):
        split_labels = labels[indices]
        split_counts = np.bincount(split_labels, minlength=2)
        if split_counts.min() == 0:
            raise RuntimeError(f"{split_name} split is missing one class. Adjust val_ratio or datasets.")

    return train_idx, val_idx, split_sizes


def subsample_dataset_stratified(full_dataset, labels, fraction, seed=42):
    """
    Subsample a dataset while maintaining class balance.
    
    Args:
        full_dataset: The full dataset object
        labels: All labels (numpy array)
        fraction: Fraction of data to keep (e.g., 0.1 for 10%)
        seed: Random seed for reproducibility
        
    Returns:
        Subset object containing the subsampled data, subsampled labels array
    """
    from torch.utils.data import Subset
    
    np.random.seed(seed)
    n_samples = len(labels)
    n_new = max(int(n_samples * fraction), 10)  # At least 10 samples
    
    # Stratified sampling - keep proportions of each class
    indices = np.arange(n_samples)
    sampled_indices = []
    
    for class_label in np.unique(labels):
        class_mask = labels == class_label
        class_indices = indices[class_mask]
        n_class_samples = max(int(len(class_indices) * fraction), 2)  # At least 2 per class
        sampled_class_indices = np.random.choice(class_indices, size=n_class_samples, replace=False)
        sampled_indices.extend(sampled_class_indices)
    
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)  # Randomize order
    
    subsampled_dataset = Subset(full_dataset, sampled_indices)
    subsampled_labels = labels[sampled_indices]
    
    return subsampled_dataset, subsampled_labels, sampled_indices


def save_run_summary(output_dir, config, best_params, best_value, test_metrics):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")
    payload = {
        "config": asdict(config),
        "best_val_accuracy": best_value,
        "best_params": best_params,
        "test_metrics": test_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def format_seconds(total_seconds):
    total_seconds = max(int(total_seconds), 0)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _infer_label_from_path(path):
    path_parts = [
        part.replace("_", " ").strip().lower()
        for part in os.path.normpath(path).split(os.sep)
    ]
    if any(part in {"no fire", "nofire"} for part in path_parts):
        return 0
    if any(part == "fire" for part in path_parts):
        return 1
    raise ValueError(f"Could not infer label from path: {path}")


def _materialize_dataset_entries(dataset):
    if isinstance(dataset, Subset):
        base_entries = _materialize_dataset_entries(dataset.dataset)
        return [base_entries[int(i)] for i in dataset.indices]
    if hasattr(dataset, "data"):
        return list(dataset.data)
    raise TypeError(f"Unsupported dataset type for integrity audit: {type(dataset)}")


def audit_dataset_integrity(train_dataset, test_dataset):
    train_entries = _materialize_dataset_entries(train_dataset)
    test_entries = _materialize_dataset_entries(test_dataset)
    train_labels = np.array([int(label) for _, _, label in train_entries], dtype=np.int64)
    test_labels = np.array([int(label) for _, _, label in test_entries], dtype=np.int64)

    train_label_errors = 0
    for rgb_path, _, label in train_entries:
        inferred = _infer_label_from_path(rgb_path)
        if int(label) != inferred:
            train_label_errors += 1

    test_label_errors = 0
    for rgb_path, _, label in test_entries:
        inferred = _infer_label_from_path(rgb_path)
        if int(label) != inferred:
            test_label_errors += 1

    train_rgb_paths = {os.path.realpath(rgb_path) for rgb_path, _, _ in train_entries}
    test_rgb_paths = {os.path.realpath(rgb_path) for rgb_path, _, _ in test_entries}
    rgb_overlap = train_rgb_paths & test_rgb_paths

    train_thermal_paths = {os.path.realpath(t_path) for _, t_path, _ in train_entries if t_path}
    test_thermal_paths = {os.path.realpath(t_path) for _, t_path, _ in test_entries if t_path}
    thermal_overlap = train_thermal_paths & test_thermal_paths

    print("\n--- Dataset Integrity Audit ---")
    print(f"Train samples: {len(train_entries)} | Test samples: {len(test_entries)}")
    print(
        "Train class counts | No Fire (0): "
        f"{int(np.sum(train_labels == 0))} | Fire (1): {int(np.sum(train_labels == 1))}"
    )
    print(
        "Test class counts  | No Fire (0): "
        f"{int(np.sum(test_labels == 0))} | Fire (1): {int(np.sum(test_labels == 1))}"
    )
    print(f"Train label mismatches: {train_label_errors} | Test label mismatches: {test_label_errors}")
    print(f"Train/Test RGB path overlap: {len(rgb_overlap)}")
    print(f"Train/Test Thermal path overlap: {len(thermal_overlap)}")

    if train_label_errors or test_label_errors:
        raise RuntimeError(
            "Label integrity check failed. Some sample labels do not match their folder names."
        )
    if rgb_overlap or thermal_overlap:
        raise RuntimeError(
            "Data leakage detected: overlapping train/test file paths were found."
        )


def create_dataloader(dataset, indices, batch_size, shuffle, labels=None, use_weighted_sampler=False):
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": use_cuda,
    }
    if use_weighted_sampler:
        if labels is None:
            raise ValueError("labels are required when use_weighted_sampler=True")
        labels = np.asarray(labels)
        class_counts = np.bincount(labels, minlength=2)
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = shuffle

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(Subset(dataset, indices), **loader_kwargs)


def _probe_batch_memory_util(dataset, sample_index, modality, batch_size, device, total_memory_bytes):
    """Run one training step and return peak memory utilization for the tested batch size."""
    model = None
    optimizer = None
    criterion = None
    supcon_criterion = None
    rgb_batch = therm_batch = labels_batch = outputs = embeddings = inputs = loss = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = prepare_model(modality, base_filters=32, dropout=0.2).to(device)
        initialize_weights(model)
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss().to(device)
        supcon_criterion = SupConLoss(temperature=0.07).to(device)

        rgb_img, therm_img, label = dataset[int(sample_index)]
        label_value = int(label.item()) if torch.is_tensor(label) else int(label)

        rgb_batch = rgb_img.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device, non_blocking=True)
        therm_batch = therm_img.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device, non_blocking=True)
        labels_batch = torch.full((batch_size,), label_value, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs, embeddings = unpack_model_outputs(forward_model(model, modality, rgb_batch, therm_batch))
        loss = criterion(outputs, labels_batch)
        if USE_SUPCON and embeddings is not None:
            loss = loss + 0.1 * supcon_criterion(embeddings, labels_batch)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)

        peak_alloc = torch.cuda.max_memory_allocated(device)
        return peak_alloc / total_memory_bytes
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return None
        raise
    finally:
        del loss, outputs, embeddings, inputs
        del rgb_batch, therm_batch, labels_batch
        del supcon_criterion, criterion, optimizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def autotune_batch_size_for_vram(dataset, train_idx, config):
    """Estimate the largest stable batch size near a target GPU VRAM utilization."""
    if not torch.cuda.is_available() or len(train_idx) == 0:
        return None

    device = torch.device("cuda")
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    target = config.target_gpu_mem_util
    max_batch_size = max(8, int(config.max_auto_batch_size))
    sample_index = int(train_idx[0])

    best = 0
    low_probe = 8
    high_probe = min(max_batch_size, 1024)

    while low_probe <= high_probe:
        util = _probe_batch_memory_util(dataset, sample_index, config.modality, low_probe, device, total_memory_bytes)
        if util is None:
            break
        if util <= target:
            best = low_probe
            low_probe *= 2
        else:
            break

    if best == 0:
        return None

    low = best + 1
    high = min(max_batch_size, low_probe - 1)
    while low <= high:
        mid = (low + high) // 2
        util = _probe_batch_memory_util(dataset, sample_index, config.modality, mid, device, total_memory_bytes)
        if util is not None and util <= target:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


def build_optimizer(model, optimizer_name, learning_rate, weight_decay):
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def compute_class_weights(labels):
    counts = np.bincount(np.asarray(labels), minlength=2)
    safe_counts = np.where(counts == 0, 1, counts)
    return counts.sum() / (2.0 * safe_counts)


def compute_metrics(all_labels, all_preds, all_probs):
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = roc_auc_score(all_labels, all_probs)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def unpack_model_outputs(model_outputs):
    if isinstance(model_outputs, tuple):
        if len(model_outputs) >= 2:
            return model_outputs[0], model_outputs[1]
        return model_outputs[0], None
    return model_outputs, None


def _to_single_channel_thermal(therm_imgs):
    return therm_imgs.mean(dim=1, keepdim=True) if therm_imgs.shape[1] != 1 else therm_imgs


def forward_model(model, modality, rgb_imgs, therm_imgs):
    normalized_modality = modality.upper()
    if FUSION_MODE not in ["EARLY", "LATE"]:
        raise ValueError(f"Invalid FUSION_MODE='{FUSION_MODE}'. Use 'EARLY' or 'LATE'.")
    therm_single = _to_single_channel_thermal(therm_imgs)

    if normalized_modality == "RGB":
        return model(rgb_imgs)
    if normalized_modality in ["THERMAL", "TIFF"]:
        return model(therm_single)
    if normalized_modality in ["RGBT", "RGB_TIFF"]:
        if FUSION_MODE == "LATE":
            return model(rgb_imgs, therm_single)
        return model(torch.cat([rgb_imgs, therm_single], dim=1))
    raise ValueError(f"Unsupported modality for forward pass: {modality}")


def train_one_epoch(model, dataloader, optimizer, ce_criterion, supcon_criterion, supcon_lambda, device, modality):
    model.train()
    running_loss = 0.0

    for rgb_imgs, therm_imgs, batch_labels in dataloader:
        rgb_imgs, therm_imgs, batch_labels = rgb_imgs.to(device), therm_imgs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        logits, embeddings = unpack_model_outputs(forward_model(model, modality, rgb_imgs, therm_imgs))
        ce_loss = ce_criterion(logits, batch_labels)
        supcon_loss = (
            supcon_criterion(embeddings, batch_labels)
            if USE_SUPCON and embeddings is not None
            else 0.0
        )
        loss = ce_loss + (supcon_lambda * supcon_loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


def train_final_model(full_dataset, train_idx, val_idx, best_params, config, labels, class_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader = create_dataloader(
        full_dataset,
        train_idx,
        best_params["batch_size"],
        shuffle=True,
        labels=labels[train_idx],
        use_weighted_sampler=True,
    )
    monitor_val_idx = val_idx
    if monitor_val_idx is None or len(monitor_val_idx) == 0:
        # Recreate a deterministic static 80/20 split for monitoring while still training on full data.
        _, monitor_val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=0.2,
            random_state=SEED,
            shuffle=True,
            stratify=labels,
        )
    use_validation = val_idx is not None and len(val_idx) > 0
    if not use_validation and monitor_val_idx is not None and len(monitor_val_idx) > 0:
        use_validation = True
    valloader = (
        create_dataloader(full_dataset, monitor_val_idx, best_params["batch_size"], shuffle=False)
        if use_validation
        else None
    )

    model = prepare_model(config.modality, best_params["base_filters"], best_params["dropout"]).to(device)
    initialize_weights(model)

    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=0.1).to(device)
    supcon_criterion = SupConLoss(temperature=best_params.get("supcon_temperature", 0.07))
    supcon_lambda = best_params.get("supcon_lambda", 0.0) if USE_SUPCON else 0.0
    optimizer = build_optimizer(model, best_params["optimizer"], best_params["learning_rate"], best_params["weight_decay"])

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = float("nan")
    best_state = copy.deepcopy(model.state_dict())
    best_train_loss = float("inf")

    for epoch in range(config.final_epochs):
        train_loss = train_one_epoch(
            model,
            trainloader,
            optimizer,
            criterion,
            supcon_criterion,
            supcon_lambda,
            device,
            config.modality,
        )
        history["train_loss"].append(train_loss)
        if use_validation:
            val_loss, val_acc, _, _, _ = evaluate(model, valloader, criterion, device, config.modality)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            if np.isnan(best_val_acc) or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
        else:
            history["val_loss"].append(float("nan"))
            history["val_acc"].append(float("nan"))
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, criterion, history, best_val_acc

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, base_filters):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class LightweightCNN(nn.Module):
    def __init__(self, in_channels, base_filters, dropout):
        super().__init__()
        self.backbone = CNNFeatureExtractor(in_channels=in_channels, base_filters=base_filters)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(base_filters * 4, 2)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.flatten(self.global_pool(feats))
        logits = self.classifier(self.dropout(feats))
        embeddings = F.normalize(feats, dim=1)
        return logits, embeddings


class GatedLateFusion(nn.Module):
    def __init__(self, base_filters, dropout):
        super().__init__()
        self.rgb_stream = CNNFeatureExtractor(in_channels=3, base_filters=base_filters)
        self.therm_stream = CNNFeatureExtractor(in_channels=1, base_filters=base_filters)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=base_filters * 4,
            num_heads=4,
            dropout=dropout,
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_filters * 4, 2),
        )

    def forward(self, rgb_imgs, therm_imgs):
        rgb_feats = self.rgb_stream(rgb_imgs)
        therm_feats = self.therm_stream(therm_imgs)

        batch_size, channels, height, width = therm_feats.shape
        sequence_length = height * width

        rgb_flat = rgb_feats.view(batch_size, channels, sequence_length).permute(2, 0, 1)
        therm_flat = therm_feats.view(batch_size, channels, sequence_length).permute(2, 0, 1)

        attn_output, _ = self.cross_attention(query=therm_flat, key=rgb_flat, value=rgb_flat)
        fused_features = therm_flat + attn_output

        fused_features = fused_features.permute(1, 2, 0).contiguous().view(batch_size, channels, height, width)
        fused_vector = self.flatten(self.global_pool(fused_features))
        return self.classifier(fused_vector)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        labels = labels.view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_anchor = mask.sum(dim=1)
        valid_anchors = positives_per_anchor > 0
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=embeddings.dtype)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positives_per_anchor + 1e-12)
        loss = -mean_log_prob_pos[valid_anchors].mean()
        return loss

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None: nn.init.zeros_(layer.bias)

def prepare_model(modality, base_filters, dropout):
    normalized_modality = modality.upper()
    if FUSION_MODE not in ["EARLY", "LATE"]:
        raise ValueError(f"Invalid FUSION_MODE='{FUSION_MODE}'. Use 'EARLY' or 'LATE'.")

    if normalized_modality in ["RGBT", "RGB_TIFF"]:
        if FUSION_MODE == "LATE":
            return GatedLateFusion(base_filters=base_filters, dropout=dropout)
        return LightweightCNN(in_channels=4, base_filters=base_filters, dropout=dropout)

    in_channels = 3 if normalized_modality == "RGB" else 1
    return LightweightCNN(in_channels=in_channels, base_filters=base_filters, dropout=dropout)

def prepare_inputs(modality, rgb_imgs, therm_imgs):
    if modality == "RGB": return rgb_imgs
    therm_single = therm_imgs.mean(dim=1, keepdim=True) if therm_imgs.shape[1] != 1 else therm_imgs
    if modality in ["TIFF", "THERMAL"]: return therm_single
    if modality in ["RGBT", "RGB_TIFF"]:
        return torch.cat([rgb_imgs, therm_single], dim=1)
    return torch.cat([rgb_imgs, therm_single], dim=1)

# ==============================================================================
# EVALUATION & PLOTTING FUNCTIONS FOR THE PAPER
# ==============================================================================
def evaluate(model, dataloader, criterion, device, modality):
    model.eval()
    total_loss, total_correct, total_items = 0.0, 0, 0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for rgb_imgs, therm_imgs, labels in dataloader:
            rgb_imgs, therm_imgs, labels = rgb_imgs.to(device), therm_imgs.to(device), labels.to(device)
            outputs, _ = unpack_model_outputs(forward_model(model, modality, rgb_imgs, therm_imgs))

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()
            total_items += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return total_loss / max(len(dataloader), 1), total_correct / max(total_items, 1), all_labels, all_probs, all_preds

def save_publication_plots(history, all_labels, all_probs, all_preds, outdir):
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Learning Curves: 1x2 subplot (Loss and Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Loss subplot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--', marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training vs. Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', color='#2ca02c', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    learning_curves_path = os.path.join(outdir, 'learning_curves.png')
    plt.savefig(learning_curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {learning_curves_path}")

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'],
                ax=ax, cbar_kws={}, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(outdir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {confusion_matrix_path}")

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='#2ca02c', linewidth=2.5, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    roc_curve_path = os.path.join(outdir, 'roc_curve.png')
    plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_curve_path}")

def save_final_results_log(
    outdir,
    config,
    best_params,
    best_val_acc,
    test_metrics,
    train_idx,
    val_idx,
    train_labels,
    test_size,
    datasets_to_use,
):
    """
    Saves comprehensive final results to a structured text log file.
    
    Args:
        outdir: Output directory path
        config: ExperimentConfig instance with datasets and modality info
        best_params: Dict of best hyperparameters from Optuna
        best_val_acc: Best validation accuracy achieved
        test_metrics: Dict with keys: precision, recall, specificity, f1, auc, tn, fp, fn, tp
        train_idx, val_idx: Indices of samples in train and validation splits
        train_labels: Labels of the train/validation pool
        test_size: Number of samples in the strict held-out test set
        datasets_to_use: Datasets used to build the training pool
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_path = os.path.join(outdir, 'final_results_log.txt')
    
    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FLAME FIRE DETECTION - FINAL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Datasets: {', '.join(config.datasets)}\n")
        f.write(f"Modality: {config.modality}\n\n")

        f.write("ABLATION SETTINGS:\n")
        f.write(f"  EVAL_MODE: {EVAL_MODE}\n")
        f.write(f"  FUSION_MODE: {FUSION_MODE}\n")
        f.write(f"  TRAIN_DATA_FRACTION: {TRAIN_DATA_FRACTION}\n")
        f.write(f"  DATASETS_TO_USE: {', '.join(datasets_to_use)}\n")
        f.write(f"  COMMON_TEST_DATASETS: {', '.join(COMMON_TEST_DATASETS)}\n\n")
        
        f.write("SPLIT STATISTICS:\n")
        f.write(f"  Train samples: {len(train_idx)}\n")
        f.write(f"  Val samples: {len(val_idx)}\n")
        f.write(f"  Test samples: {test_size}\n")
        f.write(f"  Train+Val pool size: {len(train_labels)}\n")
        f.write(f"  Total samples (Train+Val+Test): {len(train_labels) + test_size}\n\n")
        
        f.write("HYPERPARAMETERS:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"  Best Validation Accuracy: {best_val_acc:.4f}\n\n")
        
        f.write("TEST METRICS:\n")
        f.write(f"  Precision: {test_metrics.get('precision', 0):.4f}\n")
        f.write(f"  Recall: {test_metrics.get('recall', 0):.4f}\n")
        f.write(f"  Specificity: {test_metrics.get('specificity', 0):.4f}\n")
        f.write(f"  F1-Score: {test_metrics.get('f1', 0):.4f}\n")
        f.write(f"  AUC: {test_metrics.get('auc', 0):.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  True Negatives: {test_metrics.get('tn', 0)}\n")
        f.write(f"  False Positives: {test_metrics.get('fp', 0)}\n")
        f.write(f"  False Negatives: {test_metrics.get('fn', 0)}\n")
        f.write(f"  True Positives: {test_metrics.get('tp', 0)}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Saved: {log_path}")


def save_model_artifact(model, outdir):
    """Save exactly one final model artifact per run."""
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, "best_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")
    return model_path

# ==============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ==============================================================================
def build_objective(train_ds, val_ds, train_labels, class_weights, config):
    def objective(trial):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Optuna] Starting trial {trial.number + 1}/{config.optuna_trials}")

        # Hyperparameter Search Space
        optimizer_candidates = ["adamw"] if config.quick_sweep else ["adamw", "sgd"]
        lr_low, lr_high = (5e-5, 2e-3) if config.quick_sweep else (1e-5, 5e-3)

        base_filters = trial.suggest_categorical("base_filters", [8, 16, 32])
        optimizer_name = trial.suggest_categorical("optimizer", optimizer_candidates)
        learning_rate = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", list(config.batch_size_candidates))
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        supcon_temperature = trial.suggest_float("supcon_temperature", 0.05, 0.2)
        supcon_lambda = trial.suggest_float("supcon_lambda", 0.0, 1.0) if USE_SUPCON else 0.0

        print(
            f"[Optuna][Trial {trial.number + 1}] Params: "
            f"base_filters={base_filters}, opt={optimizer_name}, bs={batch_size}, lr={learning_rate:.2e}, wd={weight_decay:.2e}"
        )

        train_indices = np.arange(len(train_ds))
        val_indices = np.arange(len(val_ds))
        trainloader = create_dataloader(
            train_ds,
            train_indices,
            batch_size,
            shuffle=True,
            labels=train_labels,
            use_weighted_sampler=True,
        )
        valloader = create_dataloader(val_ds, val_indices, batch_size, shuffle=False)

        model = prepare_model(config.modality, base_filters, dropout).to(device)
        initialize_weights(model)

        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=0.1).to(device)
        supcon_criterion = SupConLoss(temperature=supcon_temperature)
        optimizer = build_optimizer(model, optimizer_name, learning_rate, weight_decay)

        val_acc = 0.0
        log_interval = max(1, config.cv_epochs // 4)
        for epoch in range(config.cv_epochs):
            train_one_epoch(
                model,
                trainloader,
                optimizer,
                criterion,
                supcon_criterion,
                supcon_lambda,
                device,
                config.modality,
            )

            _, val_acc, _, _, _ = evaluate(model, valloader, criterion, device, config.modality)

            if ((epoch + 1) % log_interval == 0) or (epoch == 0) or (epoch + 1 == config.cv_epochs):
                print(
                    f"[Optuna][Trial {trial.number + 1}/{config.optuna_trials}] "
                    f"Epoch {epoch + 1}/{config.cv_epochs} | Val acc: {val_acc:.4f}"
                )

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_acc

    return objective

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    if args.fusion_mode:
        FUSION_MODE = args.fusion_mode.upper()

    if args.list_experiments:
        print_experiment_catalog()
        raise SystemExit(0)

    config = resolve_experiment_config(args)
    if args.fusion_mode and config.modality.upper() not in ["RGBT", "RGB_TIFF"]:
        raise ValueError("--fusion-mode is only valid for multimodal modalities (RGBT or RGB_TIFF).")

    use_optuna_cv = USE_OPTUNA_CV
    if args.use_optuna_cv:
        use_optuna_cv = True
    if args.no_optuna_cv:
        use_optuna_cv = False

    set_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    datasets_to_use = list(config.datasets)
    print_run_config(config)
    print("\n--- Evaluation Controls ---")
    print(f"EVAL_MODE: {EVAL_MODE}")
    print(f"FUSION_MODE: {FUSION_MODE}")
    print(f"TRAIN_DATA_FRACTION: {TRAIN_DATA_FRACTION}")
    print(f"DATASETS_TO_USE: {datasets_to_use}")
    print(f"COMMON_TEST_DATASETS: {COMMON_TEST_DATASETS}")
    print(f"USE_OPTUNA_CV: {use_optuna_cv}")

    transform = get_default_transform(image_size=config.image_size)
    full_train_dataset = UnifiedFlameDataset(
        mode=config.modality,
        datasets=datasets_to_use,
        transform=transform,
        thermal_source=config.thermal_source,
        is_test_set=False,
    )
    labels = np.array([sample[2] for sample in full_train_dataset.data])

    if len(labels) == 0:
        raise RuntimeError("No training samples were loaded. Check dataset paths.")

    if not 0.0 < TRAIN_DATA_FRACTION <= 1.0:
        raise ValueError("TRAIN_DATA_FRACTION must be in (0, 1].")

    if TRAIN_DATA_FRACTION < 1.0:
        sampled_indices, _ = train_test_split(
            np.arange(len(labels)),
            train_size=TRAIN_DATA_FRACTION,
            random_state=SEED,
            shuffle=True,
            stratify=labels,
        )
        full_train_dataset = Subset(full_train_dataset, sampled_indices)
        labels = labels[sampled_indices]
        print(
            f"Applied Data Starvation: TRAIN_DATA_FRACTION={TRAIN_DATA_FRACTION:.3f} "
            f"-> train pool size={len(labels)}"
        )
        print(f"  Class 0: {np.sum(labels == 0)} | Class 1: {np.sum(labels == 1)}")

    def build_static_split(dataset_obj, label_array):
        if np.bincount(label_array, minlength=2).min() < 2:
            raise RuntimeError("Need at least 2 samples per class to create static train/validation split.")
        split_train_idx, split_val_idx = train_test_split(
            np.arange(len(label_array)),
            test_size=0.2,
            random_state=SEED,
            shuffle=True,
            stratify=label_array,
        )
        split_sizes_local = {"train": len(split_train_idx), "val": len(split_val_idx)}
        split_train_labels = label_array[split_train_idx]
        split_class_weights = compute_class_weights(split_train_labels)
        split_train_ds = Subset(dataset_obj, split_train_idx)
        split_val_ds = Subset(dataset_obj, split_val_idx)
        return (
            split_train_idx,
            split_val_idx,
            split_sizes_local,
            split_train_labels,
            split_class_weights,
            split_train_ds,
            split_val_ds,
        )

    # Static 80/20 Train/Validation split for Optuna (single split, no K-Fold)
    train_idx, val_idx, split_sizes, train_split_labels, class_weights, train_ds, val_ds = build_static_split(
        full_train_dataset, labels
    )

    # Handle dry run mode: subsample data and reduce iterations
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE ENABLED - Testing pipeline with subset of data")
        print("=" * 80)
        full_train_dataset, labels, subsample_indices = subsample_dataset_stratified(
            full_train_dataset, labels, args.dry_run_fraction, seed=SEED
        )
        print(f"Subsampled to {len(labels)} samples ({args.dry_run_fraction*100:.1f}% of original)")
        print(f"  Class 0: {np.sum(labels == 0)} | Class 1: {np.sum(labels == 1)}")

        # Reduce Optuna trials and epochs for faster iteration
        config = replace(config, optuna_trials=2, cv_epochs=5, final_epochs=5)
        print(f"Reduced iterations for speed: 2 Optuna trials, 5 CV epochs, 5 final epochs")
        print("=" * 80 + "\n")
        train_idx, val_idx, split_sizes, train_split_labels, class_weights, train_ds, val_ds = build_static_split(
            full_train_dataset, labels
        )

    test_ds = UnifiedFlameDataset(
        mode=config.modality,
        datasets=COMMON_TEST_DATASETS,
        transform=transform,
        thermal_source=config.thermal_source,
        is_test_set=True,
    )
    if len(test_ds) == 0:
        raise RuntimeError(
            "Held-out test set is empty. Check Willamette TEST_SET paths and modality/thermal-source settings."
        )

    audit_dataset_integrity(full_train_dataset, test_ds)

    print(
        f"Split sizes | Train: {split_sizes['train']} | "
        f"Val: {split_sizes['val']} | Test (held-out): {len(test_ds)}"
    )

    if config.auto_batch_size and torch.cuda.is_available():
        print("\n--- Auto-Tuning Batch Size for GPU VRAM ---")
        suggested_batch_size = autotune_batch_size_for_vram(full_train_dataset, train_idx, config)
        if suggested_batch_size is None:
            print(
                "Auto-tuning could not find a stable batch size at the requested target. "
                f"Keeping configured candidates: {list(config.batch_size_candidates)}"
            )
        else:
            tuned_candidates = sorted({
                max(8, suggested_batch_size // 2),
                suggested_batch_size,
            })
            config = replace(config, batch_size_candidates=tuple(tuned_candidates))
            print(
                f"Suggested max batch size near {config.target_gpu_mem_util:.0%} VRAM: {suggested_batch_size}. "
                f"Using candidates: {tuned_candidates}"
            )

    run_output_dir = os.path.join(args.output_root, config.name)
    os.makedirs(run_output_dir, exist_ok=True)

    best_params = None
    sweep_best_val_acc = float("nan")
    if use_optuna_cv:
        print("\n--- Starting Optuna Hyperparameter Optimization ---")
        optuna_start_time = time.time()

        def optuna_progress_callback(study, trial):
            completed_trials = sum(
                t.state in (
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED,
                    optuna.trial.TrialState.FAIL,
                )
                for t in study.trials
            )
            elapsed = time.time() - optuna_start_time
            avg_trial_time = elapsed / max(completed_trials, 1)
            remaining = max(config.optuna_trials - completed_trials, 0)
            eta = avg_trial_time * remaining

            trial_value = trial.value if trial.value is not None else float("nan")
            trial_duration = trial.duration.total_seconds() if trial.duration else 0.0
            best_value = study.best_value if study.best_trial is not None else float("nan")

            print(
                f"[Optuna Progress] {completed_trials}/{config.optuna_trials} trials done | "
                f"Last: {trial.state.name} value={trial_value:.4f} in {format_seconds(trial_duration)} | "
                f"Best={best_value:.4f} | Elapsed={format_seconds(elapsed)} | ETA={format_seconds(eta)}"
            )

        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        optimize_kwargs = {
            "n_trials": config.optuna_trials,
            "callbacks": [optuna_progress_callback],
        }
        try:
            study.optimize(
                build_objective(train_ds, val_ds, train_split_labels, class_weights, config),
                show_progress_bar=True,
                **optimize_kwargs,
            )
        except TypeError:
            # Older Optuna versions may not support show_progress_bar.
            study.optimize(build_objective(train_ds, val_ds, train_split_labels, class_weights, config), **optimize_kwargs)

        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (Validation Acc): {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        best_params = trial.params
        sweep_best_val_acc = float(trial.value)
    else:
        print("\n--- Skipping Optuna (USE_OPTUNA_CV=False) ---")
        best_params = copy.deepcopy(DEFAULT_HYPERPARAMS)
        if best_params["batch_size"] <= 0:
            raise ValueError("DEFAULT_HYPERPARAMS['batch_size'] must be > 0.")
        print("Using fixed default hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

    print("\n--- Training Final Model with Best Parameters ---")
    full_train_indices = np.sort(np.concatenate([train_idx, val_idx]))
    final_model, criterion, history, best_val_acc = train_final_model(
        full_train_dataset,
        full_train_indices,
        None,
        best_params,
        config,
        labels,
        class_weights,
    )
    if np.isnan(best_val_acc):
        best_val_acc = sweep_best_val_acc

    print("\n--- Final Test Evaluation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_indices = np.arange(len(test_ds))
    testloader = create_dataloader(test_ds, test_indices, best_params["batch_size"], shuffle=False)
    test_loss, test_acc, all_labels, all_probs, all_preds = evaluate(final_model, testloader, criterion, device, config.modality)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Compute comprehensive test metrics
    test_metrics = compute_metrics(all_labels, all_preds, all_probs)

    # Generate Publication Files
    print("\n--- Generating Publication Figures and Logs ---")
    save_publication_plots(history, all_labels, all_probs, all_preds, run_output_dir)
    save_final_results_log(
        run_output_dir,
        config,
        best_params,
        best_val_acc,
        test_metrics,
        train_idx,
        val_idx,
        labels,
        len(test_ds),
        datasets_to_use,
    )
    model_artifact_path = save_model_artifact(final_model, run_output_dir)
    
    # Print summary of saved outputs
    print("\n" + "=" * 80)
    print("SUMMARY OF SAVED OUTPUTS")
    print("=" * 80)
    print(f"Learning Curves:      {os.path.join(run_output_dir, 'learning_curves.png')}")
    print(f"Confusion Matrix:     {os.path.join(run_output_dir, 'confusion_matrix.png')}")
    print(f"ROC Curve:            {os.path.join(run_output_dir, 'roc_curve.png')}")
    print(f"Results Log:          {os.path.join(run_output_dir, 'final_results_log.txt')}")
    print(f"Model Artifact:       {model_artifact_path}")
    print("=" * 80)
