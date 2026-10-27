# DRY RUN MODE - Quick Pipeline Verification Guide

## Overview
Dry run mode allows you to test the entire training pipeline with a small fraction of your data, generating all output figures and logs without committing to the full ablation sweep. This is ideal for:
- Verifying that your code runs end-to-end without errors
- Checking that all figures (learning_curves.png, confusion_matrix.png, roc_curve.png) are generated correctly
- Validating that final_results_log.txt is properly formatted
- Identifying dataset or configuration issues quickly

## Quick Start

### Basic dry run (10% of data, ~5 minutes)
```bash
python train.py --experiment baseline_rgb_all --dry-run
```

### Smaller dry run (5% of data, ~2 minutes)
```bash
python train.py --experiment flame3_rgbt_tiff --dry-run --dry-run-fraction 0.05
```

### Larger dry run (25% of data, ~15 minutes)
```bash
python train.py --experiment flame1_rgb --dry-run --dry-run-fraction 0.25
```

## Flags

### `--dry-run` (boolean flag)
Enables dry run mode. When activated:
- Dataset is subsampled while maintaining class balance
- Optuna trials reduced to 2 (from 30)
- CV epochs reduced to 5 (from 20)
- Final epochs reduced to 5 (from 40)
- Clear console messages indicate dry run mode is active

### `--dry-run-fraction` (float, default: 0.1)
Specifies the fraction of data to use (0.0 to 1.0):
- `0.05` = 5% (smallest, fastest ~2 min)
- `0.1` = 10% (default, ~5 min)
- `0.25` = 25% (~15 min)
- `0.5` = 50% (~40 min)

## What Gets Generated

Regardless of data size, dry run produces the same output structure:
```
results/
└── {experiment_name}/
    ├── learning_curves.png      (1×2 subplot: loss & accuracy, 300dpi)
    ├── confusion_matrix.png     (heatmap, 300dpi)
    ├── roc_curve.png           (ROC curve with AUC, 300dpi)
    └── final_results_log.txt    (structured metrics & hyperparams)
```

## Example Workflow

1. **First verification** (5% dry run):
   ```bash
   python train.py --experiment baseline_rgb_all --dry-run --dry-run-fraction 0.05
   ```
   - Should complete in ~2 minutes
   - Check that all 4 output files are created
   - Inspect final_results_log.txt for proper formatting

2. **Larger verification** (10% dry run):
   ```bash
   python train.py --experiment baseline_rgb_all --dry-run
   ```
   - Should complete in ~5 minutes
   - Verify plots look reasonable (not just noise)
   - Check confusion matrix is balanced

3. **Full run** (when confident, remove --dry-run):
   ```bash
   python train.py --experiment baseline_rgb_all
   ```
   - This will use 100% of data and full iteration counts
   - Will take significantly longer (~1-2 hours depending on data size)

## Console Output in Dry Run Mode

When you run with `--dry-run`, you'll see:
```
================================================================================
DRY RUN MODE ENABLED - Testing pipeline with subset of data
================================================================================
Subsampled to 52 samples (5.0% of original)
  Class 0: 26 | Class 1: 26
Reduced iterations for speed: 2 Optuna trials, 5 CV epochs, 5 final epochs
================================================================================
```

## Key Differences: Dry Run vs. Full Run

| Aspect | Dry Run | Full Run |
|--------|---------|----------|
| Data Size | 5-25% | 100% |
| Optuna Trials | 2 | 30 |
| CV Epochs | 5 | 20 |
| Final Epochs | 5 | 40 |
| Runtime | 2-15 min | 1-2 hours |
| Output Quality | Lower (noise) | Higher (stable metrics) |

## Troubleshooting

**Issue: "No training samples were loaded"**
- Check that dataset paths are correct (see data_loading.py)
- Verify the experiment preset or dataset names are valid
- Try `python train.py --list-experiments` to see available presets

**Issue: Figures not generated**
- Check that --outdir directory is writable
- Verify matplotlib/seaborn are installed
- Check console for error messages about plot generation

**Issue: Out of memory error**
- Reduce --dry-run-fraction to smaller value (e.g., 0.02 for 2%)
- This shouldn't typically happen with small fractions

**Issue: Takes much longer than expected**
- If GPU is unavailable, CPU training will be slower
- Check top command to verify no other heavy processes running
- Consider using smaller --dry-run-fraction

## Tips for Efficient Testing

1. **Start small**: Always begin with 5% dry run
2. **Stack experiments**: Queue multiple dry runs with different configurations
3. **Compare sizes**: Try 5%, 10%, and 25% to understand scaling
4. **Monitor GPU**: Use `nvidia-smi` or `watch nvidia-smi` to track resource usage
5. **Save successful runs**: Keep output from successful dry runs for reference

## Next Steps After Successful Dry Run

Once dry run completes successfully:
1. Verify all 4 output files exist in results/{experiment_name}/
2. Check final_results_log.txt is properly formatted
3. Run the full ablation sweep:
   ```bash
   bash run_ablation_suite.sh
   ```
