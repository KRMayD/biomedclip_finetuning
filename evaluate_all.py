#!/usr/bin/env python3
"""
Comprehensive Evaluation Script
================================
Evaluates a checkpoint and saves detailed metrics to CSV.
"""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from config import BENIGN_PROMPTS, MALIGNANT_PROMPTS, DEFAULT_BIOMEDCLIP_CHECKPOINT
from models import load_biomedclip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--loss', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--model-checkpoint', type=str, default=DEFAULT_BIOMEDCLIP_CHECKPOINT)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_image(image_path, model, image_processor, device):
    """Encode image to embedding"""
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values)
        if hasattr(vision_outputs, 'pooler_output'):
            img_features = vision_outputs.pooler_output
        else:
            img_features = vision_outputs.last_hidden_state[:, 0, :]
        img_features = F.normalize(img_features, dim=-1)
    
    return img_features


def encode_text(texts, model, tokenizer, device):
    """Encode text to embeddings"""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        
        # Handle both tuple and BaseModelOutput formats
        if isinstance(text_outputs, tuple):
            text_outputs = text_outputs[0]
        
        if hasattr(text_outputs, 'pooler_output'):
            text_features = text_outputs.pooler_output
        elif hasattr(text_outputs, 'last_hidden_state'):
            text_features = text_outputs.last_hidden_state[:, 0, :]
        else:
            # If text_outputs is a Tensor directly
            text_features = text_outputs[:, 0, :]
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features


def get_ground_truth(image_path):
    """Extract ground truth from filename"""
    filename = Path(image_path).name
    if filename.startswith('benign'):
        return 0
    elif filename.startswith('malignant'):
        return 1
    else:
        return None


def classify_images(image_dir, model, tokenizer, image_processor, device, seed=42, pattern='*.png'):
    """Classify all images in directory"""
    image_dir = Path(image_dir)
    image_files = sorted(list(image_dir.glob(pattern)))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir} with pattern {pattern}")
    
    # Randomly select ONE prompt for each class based on seed
    np.random.seed(seed)
    benign_prompt = np.random.choice(BENIGN_PROMPTS)
    malignant_prompt = np.random.choice(MALIGNANT_PROMPTS)
    
    all_prompts = [benign_prompt, malignant_prompt]
    text_features = encode_text(all_prompts, model, tokenizer, device)
    
    # Extract single text features for each class
    benign_text_avg = text_features[0:1]
    malignant_text_avg = text_features[1:2]
    
    predictions = []
    ground_truths = []
    confidences = []
    probs_list = []
    
    for image_path in tqdm(image_files, desc="Classifying"):
        # Encode image
        img_features = encode_image(str(image_path), model, image_processor, device)
        
        # Compute similarities with averaged class embeddings
        class_features = torch.cat([benign_text_avg, malignant_text_avg], dim=0)
        similarities = (img_features @ class_features.T).squeeze(0)
        probs = F.softmax(similarities * 100, dim=0)
        
        pred_idx = similarities.argmax().item()
        confidence = probs[pred_idx].item()
        
        predictions.append(pred_idx)
        confidences.append(confidence)
        probs_list.append(probs.cpu().numpy())
        
        gt = get_ground_truth(image_path)
        ground_truths.append(gt)
    
    return {
        'predictions': np.array(predictions),
        'ground_truths': np.array(ground_truths),
        'confidences': np.array(confidences),
        'probabilities': np.array(probs_list),
        'image_files': [str(f.name) for f in image_files]
    }


def compute_metrics(predictions, ground_truths, probabilities):
    """Compute comprehensive metrics"""
    # Remove None values
    valid_idx = ground_truths != None
    predictions = predictions[valid_idx]
    ground_truths = ground_truths[valid_idx]
    probabilities = probabilities[valid_idx]
    
    if len(predictions) == 0:
        return None
    
    # Basic metrics
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, average='binary', zero_division=0)
    recall = recall_score(ground_truths, predictions, average='binary', zero_division=0)
    f1 = f1_score(ground_truths, predictions, average='binary', zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truths, predictions).ravel()
    
    # Specificity and Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall
    
    # AUC-ROC
    try:
        auc = roc_auc_score(ground_truths, probabilities[:, 1])
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'auc_roc': auc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total': len(predictions),
        'correct': int((predictions == ground_truths).sum())
    }


def save_results(results, metrics, args, results_dir):
    """Save results to CSV"""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Individual result file
    individual_file = results_dir / f"{args.exp_name}_epoch{args.epoch}.csv"
    with open(individual_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Ground Truth', 'Prediction', 'Confidence', 'Prob_Benign', 'Prob_Malignant', 'Correct'])
        
        for i, img_name in enumerate(results['image_files']):
            gt = results['ground_truths'][i]
            pred = results['predictions'][i]
            conf = results['confidences'][i]
            prob_benign = results['probabilities'][i][0]
            prob_malignant = results['probabilities'][i][1]
            correct = 'Yes' if gt == pred else 'No'
            
            gt_label = 'benign' if gt == 0 else 'malignant' if gt == 1 else 'unknown'
            pred_label = 'benign' if pred == 0 else 'malignant'
            
            writer.writerow([img_name, gt_label, pred_label, f'{conf:.4f}', 
                           f'{prob_benign:.4f}', f'{prob_malignant:.4f}', correct])
    
    # Append to combined results file
    combined_file = results_dir / 'all_results.csv'
    file_exists = combined_file.exists()
    
    with open(combined_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            # Write header
            writer.writerow([
                'Experiment', 'Loss', 'Batch_Size', 'Epoch', 'Timestamp',
                'Accuracy', 'Precision', 'Recall', 'F1_Score', 
                'Specificity', 'Sensitivity', 'AUC_ROC',
                'TP', 'TN', 'FP', 'FN', 'Total', 'Correct'
            ])
        
        # Write data
        writer.writerow([
            args.exp_name, args.loss, args.batch_size, args.epoch, 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f'{metrics["accuracy"]:.4f}',
            f'{metrics["precision"]:.4f}',
            f'{metrics["recall"]:.4f}',
            f'{metrics["f1_score"]:.4f}',
            f'{metrics["specificity"]:.4f}',
            f'{metrics["sensitivity"]:.4f}',
            f'{metrics["auc_roc"]:.4f}',
            metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn'],
            metrics['total'], metrics['correct']
        ])
    
    print(f"\n✓ Results saved to:")
    print(f"  - {individual_file}")
    print(f"  - {combined_file}")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {args.exp_name} (Epoch {args.epoch})")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test directory: {args.image_dir}")
    print(f"Device: {device}")
    print()
    
    # Load model
    model, tokenizer, image_processor = load_biomedclip(args.model_checkpoint, device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Classify images
    results = classify_images(args.image_dir, model, tokenizer, image_processor, device, seed=args.seed)
    
    # Compute metrics
    metrics = compute_metrics(
        results['predictions'],
        results['ground_truths'],
        results['probabilities']
    )
    
    if metrics is None:
        print("No valid ground truth labels found!")
        return
    
    # Print metrics
    print(f"\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}")
    print(f"  FP: {metrics['fp']}, FN: {metrics['fn']}")
    print(f"{'='*70}\n")
    
    # Save results
    save_results(results, metrics, args, args.results_dir)


if __name__ == '__main__':
    main()
