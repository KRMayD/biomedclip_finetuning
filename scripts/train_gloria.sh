#!/bin/bash
# Fine-tune BiomedCLIP with GLoRIA Loss

TRAIN_DIR="/home/jaey00ns/MedCLIP-SAMv2-main/data/breast_tumors/ttest"
OUTPUT_DIR="checkpoints/gloria_$(date +%Y%m%d_%H%M%S)"

BATCH_SIZE=8
EPOCHS=50
LR=2e-5
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=20
GPU_ID=0

echo "========================================================================"
echo "Fine-tuning BiomedCLIP with GLoRIA Loss"
echo "========================================================================"
echo "Train directory:   $TRAIN_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo "Batch size:        $BATCH_SIZE"
echo "Epochs:            $EPOCHS"
echo "Learning rate:     $LR"
echo "GPU:               $GPU_ID"
echo "========================================================================"
echo "GLoRIA: Global-Local Representations for Images"
echo "  - Global contrastive alignment"
echo "  - Local attention-based alignment"
echo "========================================================================"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --train-dir "$TRAIN_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --loss gloria \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --warmup-epochs $WARMUP_EPOCHS \
    --gpu 0

echo ""
echo "========================================================================"
echo "Training complete! Results saved to: $OUTPUT_DIR"
echo "========================================================================"
