#!/bin/bash
# Test fine-tuned BiomedCLIP model

CHECKPOINT="/home/jaey00ns/biomedclip_finetuning/experiments_all/dpo_bs80/checkpoint_epoch_5_biomedclip.pt"
IMAGE_DIR="/home/jaey00ns/MedCLIP-SAMv2-main/data/breast_tumors/ttrain"
GPU_ID=5

echo "========================================================================"
echo "Testing Fine-tuned BiomedCLIP"
echo "========================================================================"
echo "Checkpoint:   $CHECKPOINT"
echo "Image dir:    $IMAGE_DIR"
echo "GPU:          $GPU_ID"
echo "========================================================================"

CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
    --checkpoint "$CHECKPOINT" \
    --image-dir "$IMAGE_DIR" \
    --pattern "*.png" \
    --gpu 0

echo ""
echo "========================================================================"
echo "Testing complete!"
echo "========================================================================"
