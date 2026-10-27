# Quick Reference: Dry Run Commands

## Fastest Check (5% - ~2 min)
```bash
python train.py --experiment baseline_rgb_all --dry-run --dry-run-fraction 0.05
```

## Default Check (10% - ~5 min)  
```bash
python train.py --experiment baseline_rgb_all --dry-run
```

## Thorough Check (25% - ~15 min)
```bash
python train.py --experiment baseline_rgb_all --dry-run --dry-run-fraction 0.25
```

## Check specific experiment
```bash
python train.py --experiment flame3_rgbt_tiff --dry-run
```

## Full Run (disable dry run)
```bash
python train.py --experiment baseline_rgb_all
```

## Run full ablation suite (after dry run passes)
```bash
bash run_ablation_suite.sh
```

## View all experiments
```bash
python train.py --list-experiments
```

---

## What to Check After Dry Run

✓ Console outputs "SUCCESS" or similar completion message  
✓ All 4 files exist in results/{experiment_name}/:
  - learning_curves.png
  - confusion_matrix.png
  - roc_curve.png
  - final_results_log.txt

✓ final_results_log.txt contains sections for:
  - Timestamp and configuration
  - Split statistics
  - Hyperparameters
  - Test metrics (Precision, Recall, F1, AUC)
  - Confusion matrix counts

✓ PNG files are readable PNG files (not corrupt)

---

## Typical Console Output

```
--- Experiment Configuration ---
Name: baseline_rgb_all
...

================================================================================
DRY RUN MODE ENABLED - Testing pipeline with subset of data
================================================================================
Subsampled to 52 samples (5.0% of original)
  Class 0: 26 | Class 1: 26
Reduced iterations for speed: 2 Optuna trials, 5 CV epochs, 5 final epochs
================================================================================

--- Starting Optuna Hyperparameter Optimization ---

Best trial:
  Value (Mean K-Fold Val Acc): X.XXXX
  Params:
    model_name: resnetXX
    ...

--- Training Final Model with Best Parameters ---
--- Final Test Evaluation ---
...

--- Generating Publication Figures and Logs ---
Saved: results/baseline_rgb_all/learning_curves.png
Saved: results/baseline_rgb_all/confusion_matrix.png
Saved: results/baseline_rgb_all/roc_curve.png
Saved: results/baseline_rgb_all/final_results_log.txt

================================================================================
SUMMARY OF SAVED OUTPUTS
================================================================================
Learning Curves:      results/baseline_rgb_all/learning_curves.png
Confusion Matrix:     results/baseline_rgb_all/confusion_matrix.png
ROC Curve:            results/baseline_rgb_all/roc_curve.png
Results Log:          results/baseline_rgb_all/final_results_log.txt
================================================================================
```
