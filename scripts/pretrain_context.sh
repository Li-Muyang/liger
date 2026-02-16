#!/bin/bash
# Two-stage context projector pre-training pipeline
#
# Usage:
#   Stage 1 only:  bash scripts/pretrain_context.sh stage1 Beauty /path/to/backbone.pt
#   Stage 2 only:  bash scripts/pretrain_context.sh stage2 Beauty /path/to/backbone.pt /path/to/pretrain_context_proj.pt
#   Both stages:   bash scripts/pretrain_context.sh both Beauty /path/to/backbone.pt

STAGE=${1:-"both"}
DATASET_NAME=${2:-"Beauty"}
BACKBONE_PATH=${3}
CONTEXT_PROJ_PATH=${4}

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
        method.pretrain_context_proj_config.backbone_checkpoint=$BACKBONE_PATH \
        experiment_id="pretrain_ctx_${DATASET_NAME}"
fi

if [ "$STAGE" = "stage2" ] || [ "$STAGE" = "both" ]; then
    echo "========== Stage 2: Fine-tune with pre-trained context projector =========="
    # For "both", auto-discover context_proj from Stage 1 output
    if [ "$STAGE" = "both" ] || [ -z "$CONTEXT_PROJ_PATH" ]; then
        # Default: look in the Stage 1 output directory
        CONTEXT_PROJ_PATH="./results/liger/Amazon_${DATASET_NAME}/pretrain_ctx_${DATASET_NAME}_seed_42/pretrain_context_proj.pt"
    fi

    python run.py \
        dataset=amazon \
        dataset.name=$DATASET_NAME \
        seed=42 \
        device_id=0 \
        method=setting \
        test_method=liger \
        method.training_stage=finetune_with_context \
        method.pretrain_context_proj_config.backbone_checkpoint=$BACKBONE_PATH \
        method.pretrain_context_proj_path=$CONTEXT_PROJ_PATH \
        experiment_id="finetune_ctx_${DATASET_NAME}"
fi