#!/bin/bash
# Two-stage context projector pre-training pipeline
#
# Usage:
#   Stage 1 only:  bash scripts/pretrain_context.sh stage1 Beauty /path/to/backbone.pt
#   Stage 2 only:  bash scripts/pretrain_context.sh stage2 Beauty /path/to/pretrain_context_proj.pt
#   Both stages:   bash scripts/pretrain_context.sh both Beauty /path/to/backbone.pt

STAGE=${1:-"both"}
DATASET_NAME=${2:-"Beauty"}
CHECKPOINT_PATH=${3}

if [ "$STAGE" = "stage1" ] || [ "$STAGE" = "both" ]; then
    echo "========== Stage 1: Pre-train context projector =========="
    python run.py \
        dataset=amazon \
        dataset.name=$DATASET_NAME \
        seed=42 \
        device_id=0 \
        method=setting \
        test_method=liger \
        method.training_stage=pretrain_context \
        method.pretrain_context_proj_config.backbone_checkpoint=$CHECKPOINT_PATH \
        experiment_id="pretrain_ctx_${DATASET_NAME}"
fi

if [ "$STAGE" = "stage2" ] || [ "$STAGE" = "both" ]; then
    echo "========== Stage 2: Fine-tune with pre-trained context projector =========="
    # For stage2-only, CHECKPOINT_PATH is the pretrain_context_proj.pt path
    # For "both", it auto-discovers from the Stage 1 output directory
    EXTRA_ARGS=""
    if [ "$STAGE" = "stage2" ] && [ -n "$CHECKPOINT_PATH" ]; then
        EXTRA_ARGS="method.pretrain_context_proj_path=$CHECKPOINT_PATH"
    fi

    python run.py \
        dataset=amazon \
        dataset.name=$DATASET_NAME \
        seed=42 \
        device_id=0 \
        method=setting \
        test_method=liger \
        method.training_stage=finetune_with_context \
        $EXTRA_ARGS \
        experiment_id="finetune_ctx_${DATASET_NAME}"
fi