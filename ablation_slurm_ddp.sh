#!/usr/bin/env bash
#SBATCH --job-name=ablations
#SBATCH --partition=gpu
#SBATCH -w gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --output=./logs/slurm_ddp_%A_%a.out
#SBATCH --error=./logs/slurm_ddp_%A_%a.err

set -euo pipefail
set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DIC-SAM
echo "Python: $(which python)"
python -V
cd ./DIC-SAM

SCRIPT="./train/train_ddp.py"
SAVE_ROOT="./results/ablation_ddp/"
LOG_ROOT="./logs"

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

# Shared training setup.
BASE_ARGS=(
  --sam3_path ./DIC-SAM/load/sam3.pt
  --dinov3_path ./DIC-SAM/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  --dinov3_local_path ./DIC-SAM/dinov3
  --train_image_path ./DIC-SAM/data/images/train/
  --train_mask_path ./DIC-SAM/data/annotations/train/
  --val_image_path ./DIC-SAM/data/images/test/
  --val_mask_path ./DIC-SAM/data/annotations/test/
  --train_grade_csv ./DIC-SAM/train.csv
  --val_grade_csv ./DIC-SAM/val.csv
  --epochs 100
  --batch_size 4
  --num_workers 8
  --img_size 512
  --low_res_size 448
  --cluster_entropy_source pred
  --grade_class_weight 0.65 0.55 1.42 1.21 0.97 7.66
  --cluster_temperature 0.05
  --consistency_temperature 0.08
  --grade_delta 0.003
  --dino_intermediate_layer 11
  --lambda_grade_last 1.0
)


FINAL_BASELINE_FLAGS=(
  --dino_lora_rank 4
  --use_clustering true
  --cluster_num_prototypes 6
  --use_prototype_attention_pooling false
  --use_cluster_entropy true
  --lambda_proto_diversity 0.0
  --use_leaf_proxy true
  --lambda_leaf_proxy 0.1
  --use_dino_cond_refine true
  --use_cross_task_attention true
  --use_soft_roi_cluster true
  --grade_use_disease_weighted_pool true
  --roi_use_anomaly true
  --use_boundary_propagation false
  --use_consistency_correction true
  --use_interval_grade true
  --compare_last_layer_head true
  --cluster_token_source blend
  --cluster_roi_source anomaly_then_leaf
  --detach_shared_for_last_head true
)

declare -a NAMES=(
   "all"
)

declare -a EXTRA_ARGS=(
   "--use_clustering true --use_soft_roi_cluster true --roi_use_anomaly true --use_consistency_correction true --use_interval_grade true --use_cross_task_attention true --detach_shared_for_last_head true --use_dino_cond_refine true"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
NAME="${NAMES[$TASK_ID]}"

if [[ -z "${NAME:-}" ]]; then
  echo "Invalid TASK_ID=${TASK_ID}; no experiment defined."
  exit 1
fi

read -r -a EXTRA <<< "${EXTRA_ARGS[$TASK_ID]}"

echo "========================================"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Experiment: ${NAME}"
echo "Node: $(hostname)"
echo "========================================"

SLOT=$(( TASK_ID % 4 ))
GPU_0=$(( SLOT * 2 ))
GPU_1=$(( SLOT * 2 + 1 ))
export CUDA_VISIBLE_DEVICES="${GPU_0},${GPU_1}"

echo "Static GPU Allocation -> Task ${TASK_ID} strictly bound to GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

torchrun \
  --nproc_per_node=2 \
  --master_port=$((29500 + TASK_ID)) \
  "$SCRIPT" \
  "${BASE_ARGS[@]}" \
  "${FINAL_BASELINE_FLAGS[@]}" \
  --save_path "$SAVE_ROOT/$NAME" \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG_ROOT/${NAME}_ddp_${SLURM_JOB_ID}.log"

echo "Task ${TASK_ID} (${NAME}) finished."


