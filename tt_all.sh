#!/bin/bash
################################################################################
# Complete Training and Testing Automation Script
# 
# This script automatically:
# 1. Trains all loss functions with different batch sizes
# 2. Tests all checkpoints (epoch 5, 20, 50)
# 3. Saves results to CSV
# 4. Analyzes and finds best configurations
#
# Usage: bash tt_all.sh
################################################################################

set -e  # Exit on error

# Configuration
TRAIN_DIR="/home/jaey00ns/MedCLIP-SAMv2-main/data/breast_tumors/ttest"
TEST_DIR="/home/jaey00ns/MedCLIP-SAMv2-main/data/breast_tumors/ttrain"
BASE_OUTPUT_DIR="experiments_cliprefine"
EPOCHS=50
GPU_ID=5

# Loss functions to test
#LOSSES=("clip" "siglip" "hnl" "mgca" "gloria")
LOSSES=("cliprefine")
# Batch sizes to test
BATCH_SIZES=(2 4 20 40 80)

# Epochs to evaluate
EVAL_EPOCHS=(5 20 50)

# Learning rate and other hyperparameters
LR=2e-5
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=20
SEED=42

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Complete BiomedCLIP Fine-tuning Automation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Train directory: $TRAIN_DIR"
echo "  Test directory:  $TEST_DIR"
echo "  Output directory: $BASE_OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  GPU: $GPU_ID"
echo ""
echo -e "${GREEN}Loss functions:${NC} ${LOSSES[@]}"
echo -e "${GREEN}Batch sizes:${NC} ${BATCH_SIZES[@]}"
echo -e "${GREEN}Evaluation epochs:${NC} ${EVAL_EPOCHS[@]}"
echo ""
echo -e "${YELLOW}Total experiments: $((${#LOSSES[@]} * ${#BATCH_SIZES[@]}))${NC}"
echo -e "${YELLOW}Total evaluations: $((${#LOSSES[@]} * ${#BATCH_SIZES[@]} * ${#EVAL_EPOCHS[@]}))${NC}"
echo ""

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Create results directory
RESULTS_DIR="$BASE_OUTPUT_DIR/results"
mkdir -p "$RESULTS_DIR"

# Start time
START_TIME=$(date +%s)
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting experiments at $(date)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Counter for progress
TOTAL_EXPERIMENTS=$((${#LOSSES[@]} * ${#BATCH_SIZES[@]}))
CURRENT_EXP=0

# Loop through all loss functions
for LOSS in "${LOSSES[@]}"; do
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Processing Loss Function: ${LOSS}${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Loop through all batch sizes
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        CURRENT_EXP=$((CURRENT_EXP + 1))
        
        # Create experiment name
        EXP_NAME="${LOSS}_bs${BATCH_SIZE}"
        OUTPUT_DIR="$BASE_OUTPUT_DIR/${EXP_NAME}"
        
        echo ""
        echo -e "${YELLOW}────────────────────────────────────────────────────────────${NC}"
        echo -e "${YELLOW}[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] Training: ${EXP_NAME}${NC}"
        echo -e "${YELLOW}────────────────────────────────────────────────────────────${NC}"
        echo "  Loss: $LOSS"
        echo "  Batch size: $BATCH_SIZE"
        echo "  Output: $OUTPUT_DIR"
        echo ""
        
        # Skip if already trained
        if [ -f "$OUTPUT_DIR/final_model.pt" ]; then
            echo -e "${GREEN}✓ Training already completed, skipping...${NC}"
        else
            echo -e "${BLUE}Starting training...${NC}"
            
            # Run training
            CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
                --train-dir "$TRAIN_DIR" \
                --output-dir "$OUTPUT_DIR" \
                --loss "$LOSS" \
                --batch-size "$BATCH_SIZE" \
                --epochs "$EPOCHS" \
                --lr "$LR" \
                --weight-decay "$WEIGHT_DECAY" \
                --warmup-epochs "$WARMUP_EPOCHS" \
                --save-freq 5 \
                --gpu 0
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Training completed successfully${NC}"
            else
                echo -e "${RED}✗ Training failed${NC}"
                continue
            fi
        fi
        
        echo ""
        echo -e "${BLUE}Evaluating checkpoints...${NC}"
        
        # Evaluate each checkpoint
        for EPOCH in "${EVAL_EPOCHS[@]}"; do
            CHECKPOINT="$OUTPUT_DIR/checkpoint_epoch_${EPOCH}_biomedclip.pt"
            
            if [ ! -f "$CHECKPOINT" ]; then
                echo -e "${YELLOW}⚠ Checkpoint for epoch $EPOCH not found, skipping...${NC}"
                continue
            fi
            
            echo -e "${BLUE}  → Epoch $EPOCH${NC}"
            
            # Run evaluation
            CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_all.py \
                --checkpoint "$CHECKPOINT" \
                --image-dir "$TEST_DIR" \
                --exp-name "$EXP_NAME" \
                --epoch "$EPOCH" \
                --loss "$LOSS" \
                --batch-size "$BATCH_SIZE" \
                --results-dir "$RESULTS_DIR" \
                --seed "$SEED" \
                --gpu 0
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}    ✓ Evaluation completed${NC}"
            else
                echo -e "${RED}    ✗ Evaluation failed${NC}"
            fi
        done
        
        echo -e "${GREEN}✓ Completed ${EXP_NAME}${NC}"
    done
done

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}All experiments completed!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo -e "${GREEN}Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo ""

# Analyze results
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Analyzing results...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

python analyze_results.py --results-dir "$RESULTS_DIR"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}All done! Results saved to: $RESULTS_DIR${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Summary files:"
echo "  - $RESULTS_DIR/all_results.csv (all results)"
echo "  - $RESULTS_DIR/best_results.csv (best configurations)"
echo "  - $RESULTS_DIR/summary.txt (text summary)"
echo ""
