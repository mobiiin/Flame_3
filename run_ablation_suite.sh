#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/ablation_suite}"
VAL_RATIO="${VAL_RATIO:-0.2}"
TEST_RATIO="${TEST_RATIO:-0.2}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-10}"
CV_EPOCHS="${CV_EPOCHS:-20}"
FINAL_EPOCHS="${FINAL_EPOCHS:-80}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-baseline_rgb_all}"
AUTO_BATCH_SIZE="${AUTO_BATCH_SIZE:-0}"
TARGET_GPU_MEM_UTIL="${TARGET_GPU_MEM_UTIL:-0.9}"
MAX_AUTO_BATCH_SIZE="${MAX_AUTO_BATCH_SIZE:-512}"
QUICK_SWEEP="${QUICK_SWEEP:-0}"
FORCE_FRESH_RUNS="${FORCE_FRESH_RUNS:-0}"
FUSION_MODE_OVERRIDE=""
LIST_ONLY=0
if [[ -n "${BATCH_SIZE_CANDIDATES:-}" ]]; then
  IFS=' ' read -r -a BATCH_SIZE_ARRAY <<< "${BATCH_SIZE_CANDIDATES}"
else
  BATCH_SIZE_ARRAY=()
fi

DATASET_GROUPS=(
  "FLAME1"
  "FLAME2"
  "FLAME3"
  "FLAME1 FLAME2"
  "FLAME1 FLAME3"
  "FLAME2 FLAME3"
  "FLAME1 FLAME2 FLAME3"
)

MODALITY_CONFIGS=(
  "rgb|RGB|"
  "thermal_jpg|THERMAL|jpg"
  "rgbt_jpg|RGBT|jpg"
  "thermal_tiff|THERMAL|tiff"
  "rgbt_tiff|RGBT|tiff"
)

REQUESTED_RUNS=()

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$seconds"
}

render_progress_bar() {
  local completed="$1"
  local total="$2"
  local width=28
  local filled=0

  if [[ "$total" -gt 0 ]]; then
    filled=$((completed * width / total))
  fi

  local empty=$((width - filled))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')"
  bar+="$(printf '%*s' "$empty" '' | tr ' ' '-')"
  printf "[%s]" "$bar"
}

usage() {
  cat <<'EOF'
Usage: ./run_ablation_suite.sh [late|early|--fusion-mode MODE] [experiment ...]

Runs all valid dataset/modality ablations when no run names are provided.

Environment overrides:
  PYTHON_BIN      Python executable to use. Default: python
  OUTPUT_ROOT     Root output directory. Default: results/ablation_suite
  BASE_EXPERIMENT Base train.py preset to override. Default: baseline_rgb_all
  VAL_RATIO       Validation split ratio. Default: 0.2
  TEST_RATIO      Test split ratio. Default: 0.2
  OPTUNA_TRIALS   Optuna trial count. Default: 10
  CV_EPOCHS       Cross-validation epochs. Default: 8
  FINAL_EPOCHS    Final training epochs. Default: 40
  IMAGE_SIZE      Input image size. Default: 224
  AUTO_BATCH_SIZE Enable auto batch-size tuning per run (1 to enable). Default: 0
  TARGET_GPU_MEM_UTIL Target VRAM utilization when auto tuning. Default: 0.9
  MAX_AUTO_BATCH_SIZE Upper bound for auto tuning probes. Default: 512
  BATCH_SIZE_CANDIDATES Optional explicit batch sizes, e.g. "32 64 96 128"
  QUICK_SWEEP     Reduce hyperparameter sweep complexity (1 to enable). Default: 0
  FORCE_FRESH_RUNS Delete existing run folders and retrain from scratch (1 to enable). Default: 0

Examples:
  ./run_ablation_suite.sh
  ./run_ablation_suite.sh late
  ./run_ablation_suite.sh --fusion-mode early flame3__rgbt_jpg
  OPTUNA_TRIALS=10 FINAL_EPOCHS=15 ./run_ablation_suite.sh
  ./run_ablation_suite.sh flame3__rgb flame2_flame3__rgbt_jpg
  ./run_ablation_suite.sh --list
EOF
}

join_lower_underscore() {
  local parts=()
  local item
  for item in "$@"; do
    parts+=("$(printf '%s' "$item" | tr '[:upper:]' '[:lower:]')")
  done

  local joined=""
  local index
  for index in "${!parts[@]}"; do
    if [[ $index -gt 0 ]]; then
      joined+="_"
    fi
    joined+="${parts[$index]}"
  done
  printf '%s\n' "$joined"
}

supports_modality() {
  local modality_key="$1"
  shift

  local dataset
  for dataset in "$@"; do
    case "$modality_key" in
      rgb)
        ;;
      thermal_jpg|rgbt_jpg)
        [[ "$dataset" == "FLAME2" || "$dataset" == "FLAME3" ]] || return 1
        ;;
      thermal_tiff|rgbt_tiff)
        [[ "$dataset" == "FLAME3" ]] || return 1
        ;;
      *)
        return 1
        ;;
    esac
  done

  return 0
}

is_multimodal_modality_key() {
  local modality_key="$1"
  [[ "$modality_key" == rgbt_jpg || "$modality_key" == rgbt_tiff ]]
}

build_schedule() {
  local scheduled_runs=()
  local dataset_group modality_config modality_key modality thermal_source dataset_part run_name
  local datasets=()

  for dataset_group in "${DATASET_GROUPS[@]}"; do
    read -r -a datasets <<< "$dataset_group"
    dataset_part="$(join_lower_underscore "${datasets[@]}")"

    for modality_config in "${MODALITY_CONFIGS[@]}"; do
      IFS='|' read -r modality_key modality thermal_source <<< "$modality_config"
      if [[ -n "$FUSION_MODE_OVERRIDE" ]] && ! is_multimodal_modality_key "$modality_key"; then
        continue
      fi
      if supports_modality "$modality_key" "${datasets[@]}"; then
        run_name="${dataset_part}__${modality_key}"
        scheduled_runs+=("${run_name}|${dataset_group}|${modality}|${thermal_source}")
      fi
    done
  done

  printf '%s\n' "${scheduled_runs[@]}"
}

should_run() {
  local run_name="$1"
  if [[ ${#REQUESTED_RUNS[@]} -eq 0 ]]; then
    return 0
  fi

  local requested
  for requested in "${REQUESTED_RUNS[@]}"; do
    if [[ "$requested" == "$run_name" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --fusion-mode)
      if [[ $# -lt 2 ]]; then
        echo "Error: --fusion-mode requires a value (early or late)." >&2
        exit 1
      fi
      FUSION_MODE_OVERRIDE="$(printf '%s' "$2" | tr '[:lower:]' '[:upper:]')"
      shift 2
      ;;
    --fusion-mode=*)
      FUSION_MODE_OVERRIDE="$(printf '%s' "${1#*=}" | tr '[:lower:]' '[:upper:]')"
      shift
      ;;
    early|EARLY|late|LATE)
      if [[ -n "$FUSION_MODE_OVERRIDE" ]]; then
        echo "Error: fusion mode specified more than once." >&2
        exit 1
      fi
      FUSION_MODE_OVERRIDE="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
      shift
      ;;
    *)
      REQUESTED_RUNS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$FUSION_MODE_OVERRIDE" && "$FUSION_MODE_OVERRIDE" != "EARLY" && "$FUSION_MODE_OVERRIDE" != "LATE" ]]; then
  echo "Error: fusion mode must be EARLY or LATE." >&2
  exit 1
fi

if [[ "$LIST_ONLY" == "1" ]]; then
  build_schedule
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs"

mapfile -t SCHEDULE < <(build_schedule)

RUN_COUNT=0
run_entry=""
for run_entry in "${SCHEDULE[@]}"; do
  IFS='|' read -r run_name _ _ _ <<< "$run_entry"
  if should_run "$run_name"; then
    ((RUN_COUNT+=1))
  fi
done

if [[ $RUN_COUNT -eq 0 ]]; then
  echo "No matching ablation runs were selected. Use --list to see valid run names."
  exit 1
fi

echo "Running ablation suite with:"
echo "  Python: $PYTHON_BIN"
echo "  Output root: $OUTPUT_ROOT"
echo "  Base experiment: $BASE_EXPERIMENT"
if [[ -n "$FUSION_MODE_OVERRIDE" ]]; then
  echo "  Fusion mode override: $FUSION_MODE_OVERRIDE (multimodal runs only)"
else
  echo "  Fusion mode override: none (defaults to EARLY in train.py)"
fi
echo "  Validation ratio: $VAL_RATIO"
echo "  Test ratio: $TEST_RATIO"
echo "  Optuna trials: $OPTUNA_TRIALS"
echo "  CV epochs: $CV_EPOCHS"
echo "  Final epochs: $FINAL_EPOCHS"
echo "  Image size: $IMAGE_SIZE"
  echo "  Auto batch-size: $AUTO_BATCH_SIZE"
  echo "  Target GPU mem util: $TARGET_GPU_MEM_UTIL"
  echo "  Max auto batch size: $MAX_AUTO_BATCH_SIZE"
  echo "  Quick sweep: $QUICK_SWEEP"
  echo "  Force fresh runs: $FORCE_FRESH_RUNS"
  if [[ ${#BATCH_SIZE_ARRAY[@]} -gt 0 ]]; then
    echo "  Batch-size candidates override: ${BATCH_SIZE_ARRAY[*]}"
  fi
echo "  Scheduled runs: $RUN_COUNT"

SUITE_START_TS="$(date +%s)"
COMPLETED_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SKIPPED_RUNS=0
FAILED_RUN_NAMES=()
SKIPPED_RUN_NAMES=()

for run_entry in "${SCHEDULE[@]}"; do
  IFS='|' read -r run_name dataset_group modality thermal_source <<< "$run_entry"
  if ! should_run "$run_name"; then
    continue
  fi

  read -r -a datasets <<< "$dataset_group"
  log_path="$OUTPUT_ROOT/logs/${run_name}.log"
  run_output_dir="$OUTPUT_ROOT/${run_name}"

  if [[ -d "$run_output_dir" ]]; then
    if [[ "$FORCE_FRESH_RUNS" == "1" ]]; then
      echo
      echo "=== Resetting $run_name for a fresh start ==="
      echo "Removing existing output directory: $run_output_dir"
      rm -rf "$run_output_dir"
    else
      ((COMPLETED_RUNS+=1))
      ((SKIPPED_RUNS+=1))
      SKIPPED_RUN_NAMES+=("$run_name")

      NOW_TS="$(date +%s)"
      ELAPSED_SECONDS=$((NOW_TS - SUITE_START_TS))
      REMAINING_RUNS=$((RUN_COUNT - COMPLETED_RUNS))
      AVG_SECONDS_PER_RUN=$((ELAPSED_SECONDS / COMPLETED_RUNS))
      ETA_SECONDS=$((AVG_SECONDS_PER_RUN * REMAINING_RUNS))
      PERCENT_DONE=$((COMPLETED_RUNS * 100 / RUN_COUNT))

      echo
      echo "=== Skipping $run_name ==="
      echo "Reason: output directory already exists at $run_output_dir"
      echo "Overall progress: $(render_progress_bar "$COMPLETED_RUNS" "$RUN_COUNT") ${PERCENT_DONE}% (${COMPLETED_RUNS}/${RUN_COUNT})"
      echo "Elapsed: $(format_duration "$ELAPSED_SECONDS") | Remaining runs: $REMAINING_RUNS | ETA: $(format_duration "$ETA_SECONDS")"
      continue
    fi
  fi

  cmd=(
    "$PYTHON_BIN" train.py
    --experiment "$BASE_EXPERIMENT"
    --run-name "$run_name"
    --datasets "${datasets[@]}"
    --modality "$modality"
    --output-root "$OUTPUT_ROOT"
    --val-ratio "$VAL_RATIO"
    --test-ratio "$TEST_RATIO"
    --optuna-trials "$OPTUNA_TRIALS"
    --cv-epochs "$CV_EPOCHS"
    --final-epochs "$FINAL_EPOCHS"
    --image-size "$IMAGE_SIZE"
    --target-gpu-mem-util "$TARGET_GPU_MEM_UTIL"
    --max-auto-batch-size "$MAX_AUTO_BATCH_SIZE"
  )
  if [[ ${#BATCH_SIZE_ARRAY[@]} -gt 0 ]]; then
    cmd+=(--batch-size-candidates "${BATCH_SIZE_ARRAY[@]}")
  fi
  if [[ "$AUTO_BATCH_SIZE" == "1" ]]; then
    cmd+=(--auto-batch-size)
  fi
  if [[ "$QUICK_SWEEP" == "1" ]]; then
    cmd+=(--quick-sweep)
  fi
  if [[ -n "$thermal_source" ]]; then
    cmd+=(--thermal-source "$thermal_source")
  fi
  if [[ -n "$FUSION_MODE_OVERRIDE" ]]; then
    cmd+=(--fusion-mode "$FUSION_MODE_OVERRIDE")
  fi

  echo
  echo "=== Running $run_name ==="
  echo "Progress before run: $(render_progress_bar "$COMPLETED_RUNS" "$RUN_COUNT") (${COMPLETED_RUNS}/${RUN_COUNT})"
  echo "Datasets: $dataset_group"
  echo "Modality: $modality"
  if [[ -n "$thermal_source" ]]; then
    echo "Thermal source: $thermal_source"
  fi
  echo "Log: $log_path"

  RUN_START_TS="$(date +%s)"

  run_status=0
  if "${cmd[@]}" 2>&1 | tee "$log_path"; then
    run_status=0
  else
    run_status=$?
  fi

  ((COMPLETED_RUNS+=1))
  RUN_END_TS="$(date +%s)"
  RUN_DURATION=$((RUN_END_TS - RUN_START_TS))
  ELAPSED_SECONDS=$((RUN_END_TS - SUITE_START_TS))
  REMAINING_RUNS=$((RUN_COUNT - COMPLETED_RUNS))
  AVG_SECONDS_PER_RUN=$((ELAPSED_SECONDS / COMPLETED_RUNS))
  ETA_SECONDS=$((AVG_SECONDS_PER_RUN * REMAINING_RUNS))
  PERCENT_DONE=$((COMPLETED_RUNS * 100 / RUN_COUNT))

  echo
  if [[ $run_status -eq 0 ]]; then
    ((SUCCESSFUL_RUNS+=1))
    echo "Run completed: $run_name"
  else
    ((FAILED_RUNS+=1))
    FAILED_RUN_NAMES+=("$run_name")
    echo "Run failed: $run_name (exit code: $run_status)"
    echo "  See log: $log_path"
  fi
  echo "  Run duration: $(format_duration "$RUN_DURATION")"
  echo "  Overall progress: $(render_progress_bar "$COMPLETED_RUNS" "$RUN_COUNT") ${PERCENT_DONE}% (${COMPLETED_RUNS}/${RUN_COUNT})"
  echo "  Elapsed: $(format_duration "$ELAPSED_SECONDS") | Remaining runs: $REMAINING_RUNS | ETA: $(format_duration "$ETA_SECONDS")"
done

echo
SUITE_END_TS="$(date +%s)"
TOTAL_SUITE_SECONDS=$((SUITE_END_TS - SUITE_START_TS))
echo "Ablation suite completed. Results are under $OUTPUT_ROOT"
echo "Total suite time: $(format_duration "$TOTAL_SUITE_SECONDS")"
echo "Successful runs: $SUCCESSFUL_RUNS"
echo "Failed runs: $FAILED_RUNS"
echo "Skipped runs: $SKIPPED_RUNS"

if [[ $SKIPPED_RUNS -gt 0 ]]; then
  echo "Skipped run names: ${SKIPPED_RUN_NAMES[*]}"
fi

if [[ $FAILED_RUNS -gt 0 ]]; then
  echo "Failed run names: ${FAILED_RUN_NAMES[*]}"
  exit 1
fi
