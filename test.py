#!/usr/bin/env python3
"""
Test Script for Fine-tuned BiomedCLIP
=====================================
Classify tumors as benign or malignant using the fine-tuned model.

Usage:
    # Single image
    python test.py --checkpoint checkpoints/clip_exp/best_model.pt --image path/to/image.png
    
    # Batch inference
    python test.py --checkpoint checkpoints/clip_exp/best_model.pt --image-dir path/to/images
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import BENIGN_PROMPTS, MALIGNANT_PROMPTS, DEFAULT_BIOMEDCLIP_CHECKPOINT
from models import load_biomedclip


def parse_args():
    parser = argparse.ArgumentParser(description="Test fine-tuned BiomedCLIP")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path to classify')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory of images to classify')
    parser.add_argument('--pattern', type=str, default='*.png',
                        help='File pattern for batch classification')
    parser.add_argument('--model-checkpoint', type=str,
                        default=DEFAULT_BIOMEDCLIP_CHECKPOINT,
                        help='Base BiomedCLIP checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-simple-prompts', action='store_true',
                        help='Use simple "benign tumor"/"malignant tumor" prompts instead of detailed ones')
    
    return parser.parse_args()


def set_seed(seed):
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
        
        if hasattr(text_outputs, 'pooler_output'):
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs.last_hidden_state[:, 0, :]
        
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features


def classify_tumor(image_path, model, tokenizer, image_processor, device, use_simple=False):
    """Classify tumor as benign or malignant"""
    
    if use_simple:
        # Simple prompts
        benign_prompt = "benign tumor"
        malignant_prompt = "malignant tumor"
    else:
        # Detailed prompts (same as training)
        benign_prompt = random.choice(BENIGN_PROMPTS)
        malignant_prompt = random.choice(MALIGNANT_PROMPTS)
    
    prompts = [benign_prompt, malignant_prompt]
    class_names = ["benign", "malignant"]
    
    # Encode image and text
    img_features = encode_image(image_path, model, image_processor, device)
    text_features = encode_text(prompts, model, tokenizer, device)
    
    # Calculate similarities
    similarities = (img_features @ text_features.T).squeeze(0)
    probs = F.softmax(similarities * 100, dim=0)  # Temperature scaling
    
    # Get prediction
    pred_idx = similarities.argmax().item()
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()
    
    return {
        'prediction': pred_class,
        'confidence': confidence,
        'probabilities': {
            'benign': probs[0].item(),
            'malignant': probs[1].item()
        },
        'similarities': {
            'benign': similarities[0].item(),
            'malignant': similarities[1].item()
        }
    }


def get_ground_truth(image_path):
    """Extract ground truth from filename"""
    filename = Path(image_path).name
    if filename.startswith('benign'):
        return 'benign'
    elif filename.startswith('malignant'):
        return 'malignant'
    else:
        return None


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Load model
    print(f"[INFO] Loading model from: {args.checkpoint}")
    model, tokenizer, image_processor = load_biomedclip(args.model_checkpoint, device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("[INFO] Model loaded successfully")
    
    # Single image classification
    if args.image:
        print(f"\n[INFO] Classifying image: {args.image}")
        result = classify_tumor(args.image, model, tokenizer, image_processor, device, args.use_simple_prompts)
        
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        print(f"  Benign:    {result['probabilities']['benign']:.2%}")
        print(f"  Malignant: {result['probabilities']['malignant']:.2%}")
        print(f"\nSimilarities:")
        print(f"  Benign:    {result['similarities']['benign']:.4f}")
        print(f"  Malignant: {result['similarities']['malignant']:.4f}")
        
        # Check ground truth if available
        gt = get_ground_truth(args.image)
        if gt:
            correct = "✓" if result['prediction'] == gt else "✗"
            print(f"\nGround Truth: {gt} {correct}")
    
    # Batch classification
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = list(image_dir.glob(args.pattern))
        
        if len(image_files) == 0:
            print(f"[ERROR] No images found in {image_dir} with pattern {args.pattern}")
            return
        
        print(f"\n[INFO] Classifying {len(image_files)} images from {image_dir}")
        
        correct = 0
        total = 0
        results = []
        
        for image_path in tqdm(image_files, desc="Classifying"):
            result = classify_tumor(str(image_path), model, tokenizer, image_processor, device, args.use_simple_prompts)
            gt = get_ground_truth(image_path)
            
            if gt:
                total += 1
                if result['prediction'] == gt:
                    correct += 1
            
            results.append({
                'image': image_path.name,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'ground_truth': gt,
                'correct': result['prediction'] == gt if gt else None
            })
        
        # Print results
        print(f"\n{'='*80}")
        print(f"{'Image':<40} {'Prediction':<12} {'Confidence':<12} {'GT':<10} {'Result'}")
        print(f"{'='*80}")
        
        for r in results:
            gt_str = r['ground_truth'] if r['ground_truth'] else 'N/A'
            result_str = "✓" if r['correct'] else ("✗" if r['correct'] is not None else "-")
            print(f"{r['image']:<40} {r['prediction']:<12} {r['confidence']:<12.2%} {gt_str:<10} {result_str}")
        
        # Print accuracy if ground truth available
        if total > 0:
            accuracy = correct / total
            print(f"\n{'='*80}")
            print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
            print(f"{'='*80}")
    
    else:
        print("[ERROR] Please specify either --image or --image-dir")


if __name__ == '__main__':
    main()
