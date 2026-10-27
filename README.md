# FLAME fire detection experiments

This repository trains and evaluates fire versus no-fire image classifiers on FLAME 1, FLAME 2, and FLAME 3 data. It supports RGB images, thermal JPG or TIFF images, and paired RGB plus thermal inputs. The ablation script runs supported dataset and modality combinations with optional early or late fusion for paired inputs.

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate flame3_env
```

The datasets are not included in this repository. In [`data_loading.py`](data_loading.py), set `BASE_DATA_DIR` to the directory containing `Flame 1`, `Flame 2`, and `Flame 3`. Check the `DATASET_PATHS` entries against your local folder layout, including the separate FLAME 3 `TEST_SET` folders. RGB and thermal files must have matching names or frame IDs for paired runs.

## Run an experiment

List the built-in presets:

```bash
python train.py --list-experiments
```

Run a small pipeline check, then a full experiment:

```bash
python train.py --experiment flame3_rgbt_tiff --dry-run --dry-run-fraction 0.05
python train.py --experiment flame3_rgbt_tiff
```

`train.py` uses the default hyperparameters unless `--use-optuna-cv` is supplied. Add `--fusion-mode LATE` to change fusion for a paired RGB and thermal run. See [`DRY_RUN_GUIDE.md`](DRY_RUN_GUIDE.md) for more dry-run options.

## Run the ablation suite

```bash
./run_ablation_suite.sh --list
./run_ablation_suite.sh flame3__rgbt_jpg
./run_ablation_suite.sh --fusion-mode late flame3__rgbt_jpg
./run_ablation_suite.sh
```

With no run names, the script runs every supported dataset and modality combination. It skips a run when its output directory already exists. Set `OUTPUT_ROOT` to change the destination, or `FORCE_FRESH_RUNS=1` to retrain existing runs. Run `./run_ablation_suite.sh --help` for all options.

## Outputs

By default, each training run writes to `results/<run_name>/`:

- `best_model.pt`: saved model weights
- `learning_curves.png`, `confusion_matrix.png`, and `roc_curve.png`: evaluation plots
- `final_results_log.txt`: configuration and metrics

The suite also writes console logs under `results/ablation_suite/logs/`. The `results/` directory is ignored by Git, so generated artifacts stay local.

## Files

- [`train.py`](train.py): experiment configuration, training, evaluation, and output generation
- [`data_loading.py`](data_loading.py): dataset paths, labels, and RGB/thermal pairing
- [`run_ablation_suite.sh`](run_ablation_suite.sh): batch experiment runner
- [`align_rgb_thermal_pairs.py`](align_rgb_thermal_pairs.py): RGB/thermal alignment utility
- [`prepare_flame2_subsets.py`](prepare_flame2_subsets.py): FLAME 2 subset preparation
- [`rename_results_artifacts.py`](rename_results_artifacts.py): result artifact naming utility
