# BiomedCLIP Fine-tuning

Quick experiments with different contrastive losses for BiomedCLIP fine-tuning.

This project uses the MedCLIP-SAMv2 virtual environment. Make sure to activate it before running.

## What's this?

Fine-tune BiomedCLIP with various contrastive loss functions:
- **CLIP** - Standard InfoNCE
- **SigLIP** - Sigmoid-based pairwise loss
- **HNL** - Hard Negative Loss with reweighting
- **MGCA** - Multi-Granularity Cross-modal Alignment
- **GLoRIA** - Global-Local attention-based learning

Single training script, different loss functions via `--loss` flag.

## Usage

**Train:**
```bash
python train.py \
    --train-dir /path/to/images \
    --output-dir checkpoints/my_exp \
    --loss clip \
    --batch-size 8 \
    --epochs 50 \
    --gpu 0
```

Change `--loss` to: `clip`, `siglip`, `hnl`, `mgca`, or `gloria`

**Test:**
```bash
python test.py \
    --checkpoint checkpoints/my_exp/best_model.pt \
    --image-dir /path/to/test_images \
    --gpu 0
```

**Run all experiments:**
```bash
bash tt_all.sh
```
Tests all losses with different batch sizes, saves results to CSV.

## Structure

```
losses/          # Loss implementations (clip, siglip, hnl, mgca, gloria)
models/          # BiomedCLIP wrappers
data/            # Dataset loader
train.py         # Main training script
test.py          # Evaluation script
tt_all.sh        # Run all experiments automatically
config.py        # Hyperparameters and prompts
```

## Adding New Loss

1. Create `losses/my_loss.py`
2. Add to `losses/__init__.py`
3. Add config to `config.py`
4. Update `train.py` create_loss_function()

## Notes

- Dataset format: images named `benign*.png` or `malignant*.png`
- Saves checkpoints every 5 epochs
- TensorBoard logs in `checkpoints/<exp>/logs/`
- Adjust batch size based on GPU memory
