#!/usr/bin/env python3
"""
Unified Training Script for BiomedCLIP Fine-tuning
==================================================
Supports multiple loss functions: CLIP, SigLIP, HNL
Usage:
    python train.py --loss clip --train-dir /path/to/train --output-dir checkpoints/clip_exp
"""
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[WARNING] TensorBoard not available. Logging to console only.")

from config import DEFAULT_BIOMEDCLIP_CHECKPOINT, LOSS_CONFIGS, DEFAULT_TRAINING_CONFIG
from models import load_model, get_biomedclip_features, get_biomedclip_features_mgca
from data import TumorDataset, CsvCaptionDataset
from losses import CLIPLoss, SigLIPLoss, HardNegativeLoss, MGCALoss, GLoRIALoss, CLIPRefineLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BiomedCLIP with various losses")
    
    # Data
    parser.add_argument('--train-dir', type=str, required=False, 
                        help='Training images directory (not required for DPO with CSV)')
    parser.add_argument('--train-csv', type=str, default=None,
                        help='CSV file containing image-caption pairs for non-DPO losses')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Validation images directory (optional)')
    parser.add_argument('--val-csv', type=str, default=None,
                        help='Validation CSV file containing image-caption pairs (optional)')
    parser.add_argument('--csv-img-key', type=str, default='filename',
                        help='Image path column for --train-csv/--val-csv')
    parser.add_argument('--csv-caption-key', type=str, default='Caption',
                        help='Caption column for --train-csv/--val-csv')
    parser.add_argument('--csv-separator', type=str, default=',',
                        help='CSV separator for --train-csv/--val-csv')
    parser.add_argument('--csv-image-root', type=str, default=None,
                        help='Optional root for relative image paths in CSV')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Optional maximum number of samples for quick smoke tests')
    
    # Model
    parser.add_argument('--model-checkpoint', type=str, 
                        default=DEFAULT_BIOMEDCLIP_CHECKPOINT,
                        help='Hugging Face BiomedCLIP checkpoint or local OpenCLIP state dict')
    parser.add_argument('--model-backend', choices=['biomedclip', 'openclip'], default='biomedclip',
                        help='Model loader backend')
    parser.add_argument('--openclip-model', type=str, default='ViT-B-32-quickgelu',
                        help='OpenCLIP architecture when --model-backend=openclip')
    parser.add_argument('--openclip-src', type=str,
                        default='/home/msko021220/finegrained-vlm-training/biomedclip_finetuning/open_clip/src',
                        help='Local OpenCLIP source directory when --model-backend=openclip')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='clip',
                        choices=['clip', 'siglip', 'hnl', 'mgca', 'gloria', 'dpo', 'cliprefine'],
                        help='Loss function to use')
    parser.add_argument('--disable-embedding-normalization', action='store_true',
                        help='Use raw image/text embeddings for CLIP or CLIPRefine contrastive logits.')
    
    # DPO specific arguments
    parser.add_argument('--dpo-csv', type=str, default=None,
                        help='CSV file for DPO training (required if using DPO loss)')
    parser.add_argument('--neg-dir', type=str, default=None,
                        help='Negative images directory for DPO (deprecated, use --dpo-csv instead)')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=DEFAULT_TRAINING_CONFIG['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAINING_CONFIG['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULT_TRAINING_CONFIG['lr'])
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_TRAINING_CONFIG['weight_decay'])
    parser.add_argument('--warmup-epochs', type=int, default=DEFAULT_TRAINING_CONFIG['warmup_epochs'])
    parser.add_argument('--num-workers', type=int, default=DEFAULT_TRAINING_CONFIG['num_workers'])
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=DEFAULT_TRAINING_CONFIG['seed'])
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loss_function(loss_name, normalize_features=True):
    """Create loss function based on name"""
    config = LOSS_CONFIGS[loss_name]
    
    if loss_name == 'clip':
        return CLIPLoss(
            temperature=config['temperature'],
            normalize_features=normalize_features,
        )
    elif loss_name == 'siglip':
        return SigLIPLoss(temperature=config['temperature'], bias=config['bias'])
    elif loss_name == 'hnl':
        return HardNegativeLoss(
            temperature=config['temperature'],
            beta1=config['beta1'],
            beta2=config['beta2'],
            alpha=config['alpha']
        )
    elif loss_name == 'mgca':
        return MGCALoss(
            input_dim=768,
            emb_dim=config['emb_dim'],
            hidden_dim=config['hidden_dim'],
            num_prototypes=config['num_prototypes'],
            softmax_temperature=config['softmax_temperature'],
            local_temperature=config['local_temperature'],
            proto_temperature=config['proto_temperature'],
            lambda_1=config['lambda_1'],
            lambda_2=config['lambda_2'],
            lambda_3=config['lambda_3'],
            epsilon=config['epsilon'],
            sinkhorn_iterations=config['sinkhorn_iterations'],
            bidirectional=config['bidirectional'],
            use_local_atten=config['use_local_atten'],
            num_heads=config['num_heads']
        )
    elif loss_name == 'gloria':
        return GLoRIALoss(
            temp_global=config['temp_global'],
            temp_local1=config['temp_local1'],
            temp_local2=config['temp_local2'],
            temp_local3=config['temp_local3'],
            lambda_global=config['lambda_global'],
            lambda_local=config['lambda_local'],
            local_agg=config['local_agg']
        )
    elif loss_name == 'dpo':
        from losses import DPOLoss
        return DPOLoss(
            alpha=config['alpha'],
            beta=config['beta']
        )
    elif loss_name == 'cliprefine':
        return CLIPRefineLoss(
            temperature=config['temperature'],
            lambda_rand=config['lambda_rand'],
            strategy=config['strategy'],
            share_random_feat=config['share_random_feat'],
            mu=config['mu'],
            sigma=config['sigma'],
            precomputed_stats=config['precomputed_stats'],
            regularization_decay=config['regularization_decay'],
            max_iteration=config['max_iteration'],
            normalize_features=normalize_features,
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer=None, use_local_features=False, ref_model=None):
    """Train for one epoch"""
    model.train()
    if ref_model is not None:
        ref_model.eval()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # DPO mode
        if ref_model is not None:
            # Move to device
            imgs_pos = batch['imgs_pos'].to(device)
            caption_ids_pos = batch['caption_ids_pos'].to(device)
            attention_mask_pos = batch['attention_mask_pos'].to(device)
            imgs_neg = batch['imgs_neg'].to(device)
            caption_ids_neg = batch['caption_ids_neg'].to(device)
            attention_mask_neg = batch['attention_mask_neg'].to(device)
            
            # Policy model forward (positive)
            batch_pos = {
                'imgs': imgs_pos,
                'caption_ids': caption_ids_pos,
                'attention_mask': attention_mask_pos
            }
            img_feat_pos, txt_feat_pos = get_biomedclip_features(model, batch_pos)
            
            # Policy model forward (negative)
            batch_neg = {
                'imgs': imgs_neg,
                'caption_ids': caption_ids_neg,
                'attention_mask': attention_mask_neg
            }
            img_feat_neg, txt_feat_neg = get_biomedclip_features(model, batch_neg)
            
            # Reference model forward (no grad)
            with torch.no_grad():
                ref_img_feat_pos, ref_txt_feat_pos = get_biomedclip_features(ref_model, batch_pos)
                ref_img_feat_neg, ref_txt_feat_neg = get_biomedclip_features(ref_model, batch_neg)
            
            # Compute DPO loss
            loss = criterion(
                img_feat_pos, txt_feat_pos,
                img_feat_neg, txt_feat_neg,
                ref_img_feat_pos, ref_txt_feat_pos,
                ref_img_feat_neg, ref_txt_feat_neg
            )
        else:
            # Standard mode
            # Move to device
            imgs = batch['imgs'].to(device)
            caption_ids = batch['caption_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_on_device = {
                'imgs': imgs,
                'caption_ids': caption_ids,
                'attention_mask': attention_mask
            }
            
            # Forward pass
            if use_local_features:
                # MGCA and GLoRIA require global and local features
                image_features_dict, text_features_dict = get_biomedclip_features_mgca(model, batch_on_device)
                loss, loss_dict = criterion(image_features_dict, text_features_dict)
            else:
                # Simple losses only need global features
                image_features, text_features = get_biomedclip_features(model, batch_on_device)
                loss = criterion(image_features, text_features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Logging
        total_loss += loss.item()
        if ref_model is not None:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        elif use_local_features:
            postfix = {
                'loss': f'{loss.item():.4f}',
                'global': f"{loss_dict['loss_global']:.4f}",
                'local': f"{loss_dict['loss_local']:.4f}",
            }
            # Add proto loss only if it exists (MGCA has it, GLoRIA doesn't)
            if 'loss_proto' in loss_dict:
                postfix['proto'] = f"{loss_dict['loss_proto']:.4f}"
            progress_bar.set_postfix(postfix)
        else:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # TensorBoard logging
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            if use_local_features and ref_model is None:
                writer.add_scalar('train/loss_global', loss_dict['loss_global'], global_step)
                writer.add_scalar('train/loss_local', loss_dict['loss_local'], global_step)
                if 'loss_proto' in loss_dict:
                    writer.add_scalar('train/loss_proto', loss_dict['loss_proto'], global_step)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            imgs = batch['imgs'].to(device)
            caption_ids = batch['caption_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_on_device = {
                'imgs': imgs,
                'caption_ids': caption_ids,
                'attention_mask': attention_mask
            }
            
            image_features, text_features = get_biomedclip_features(model, batch_on_device)
            loss = criterion(image_features, text_features)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    args = parse_args()
    
    # Validate arguments based on loss type
    if args.loss == 'dpo':
        if args.dpo_csv is None and args.neg_dir is None:
            raise ValueError("Either --dpo-csv or --neg-dir is required when using DPO loss")
        if args.dpo_csv is not None and args.neg_dir is not None:
            raise ValueError("Cannot use both --dpo-csv and --neg-dir. Please use --dpo-csv for CSV-based DPO training")
    else:
        # For non-DPO losses, either --train-csv or --train-dir is required.
        if args.train_csv is None and args.train_dir is None:
            raise ValueError(f"Either --train-csv or --train-dir is required when using {args.loss} loss")
        if args.train_csv is not None and args.train_dir is not None:
            raise ValueError("Use only one of --train-csv or --train-dir for non-DPO training")
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        log_dir = output_dir / 'logs'
        writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs: {log_dir}")
    
    # Load model
    model, tokenizer, processor = load_model(
        backend=args.model_backend,
        checkpoint=args.model_checkpoint,
        device=device,
        openclip_model=args.openclip_model,
        openclip_src=args.openclip_src,
    )
    
    # Load reference model for DPO
    ref_model = None
    if args.loss == 'dpo':
        print("[INFO] Loading reference model for DPO...")
        ref_checkpoint = LOSS_CONFIGS['dpo']['ref_checkpoint']
        
        # Load reference model
        ref_model, _, _ = load_model(
            backend=args.model_backend,
            checkpoint=args.model_checkpoint,
            device=device,
            openclip_model=args.openclip_model,
            openclip_src=args.openclip_src,
        )
        
        # Load reference weights
        if ref_checkpoint.endswith('.bin'):
            # Load HuggingFace format weights
            state_dict = torch.load(ref_checkpoint, map_location=device)
            # Load into ref_model (this may need conversion depending on format)
            try:
                ref_model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded reference weights from {ref_checkpoint}")
            except Exception as e:
                print(f"[WARNING] Could not load reference weights directly: {e}")
                print(f"[INFO] Attempting to load into model components...")
                # Try loading into vision and text models separately
                try:
                    ref_model.vision_model.load_state_dict(state_dict, strict=False)
                    ref_model.text_model.load_state_dict(state_dict, strict=False)
                    print(f"[INFO] Loaded reference weights into vision/text models")
                except Exception as e2:
                    print(f"[ERROR] Failed to load reference weights: {e2}")
                    print(f"[INFO] Using pretrained weights as reference")
        
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("[INFO] Reference model initialized and frozen")
    
    # Create datasets
    if args.loss == 'dpo':
        # DPO dataset with positive/negative pairs
        if args.dpo_csv is not None:
            # CSV-based DPO dataset
            from data.dpo_tumor_dataset import DPOTumorDataset
            train_dataset = DPOTumorDataset(args.dpo_csv, processor, tokenizer)
        else:
            # Legacy: directory-based DPO dataset
            from data.tumor_dataset import DPOTumorDataset
            train_dataset = DPOTumorDataset(args.train_dir, args.neg_dir, processor, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        # Standard dataset
        if args.train_csv:
            train_dataset = CsvCaptionDataset(
                args.train_csv,
                processor,
                tokenizer,
                image_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                separator=args.csv_separator,
                image_root=args.csv_image_root,
                max_length=DEFAULT_TRAINING_CONFIG['max_length'],
                max_samples=args.max_samples,
            )
        else:
            train_dataset = TumorDataset(args.train_dir, processor, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    val_loader = None
    if args.val_csv or args.val_dir:
        if args.val_csv and args.val_dir:
            raise ValueError("Use only one of --val-csv or --val-dir")
        if args.val_csv:
            val_dataset = CsvCaptionDataset(
                args.val_csv,
                processor,
                tokenizer,
                image_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                separator=args.csv_separator,
                image_root=args.csv_image_root,
                max_length=DEFAULT_TRAINING_CONFIG['max_length'],
                max_samples=args.max_samples,
            )
        else:
            val_dataset = TumorDataset(args.val_dir, processor, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create loss function
    criterion = create_loss_function(
        args.loss,
        normalize_features=not args.disable_embedding_normalization,
    ).to(device)
    print(f"[INFO] Using loss function: {args.loss}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    print(f"[INFO] Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    best_val_loss = float('inf')
    use_local_features = (args.loss in ['mgca', 'gloria'])  # Both need local features
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, writer, use_local_features, ref_model)
        print(f"\n[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f}")
        
        if writer:
            writer.add_scalar('train/loss_epoch', train_loss, epoch)
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"[Epoch {epoch}/{args.epochs}] Val Loss: {val_loss:.4f}")
            
            if writer:
                writer.add_scalar('val/loss', val_loss, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, output_dir / 'best_model.pt')
                print(f"[INFO] Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}_biomedclip.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\n[INFO] Training complete! Final model saved to: {final_path}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
