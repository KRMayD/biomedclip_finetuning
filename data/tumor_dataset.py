#!/usr/bin/env python3
"""
Dataset for Tumor Classification
=================================
"""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from config import BENIGN_PROMPTS, MALIGNANT_PROMPTS


class TumorDataset(Dataset):
    """
    Breast Tumor Dataset with random prompts.
    
    Args:
        image_dir (str): Directory containing tumor images
        processor: Image processor from BiomedCLIP
        tokenizer: Text tokenizer from BiomedCLIP
        max_length (int): Maximum text length for tokenization
    """
    def __init__(self, image_dir, processor, tokenizer, max_length=77):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Build image list
        self.samples = []
        for img_path in self.image_dir.glob('*.png'):
            img_name = img_path.name
            if img_name.startswith('benign'):
                label = 'benign'
            elif img_name.startswith('malignant'):
                label = 'malignant'
            else:
                continue
            
            self.samples.append({
                'image_path': img_path,
                'label': label
            })
        
        print(f"[INFO] Loaded {len(self.samples)} samples from {image_dir}")
        benign_count = sum(1 for s in self.samples if s['label'] == 'benign')
        malignant_count = sum(1 for s in self.samples if s['label'] == 'malignant')
        print(f"       Benign: {benign_count}, Malignant: {malignant_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Process image
        if callable(self.processor):
            processed = self.processor(image)
            if isinstance(processed, dict):
                pixel_values = processed['pixel_values'][0]
            else:
                pixel_values = processed
        else:
            pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        
        # Random caption
        if sample['label'] == 'benign':
            caption = random.choice(BENIGN_PROMPTS)
        else:
            caption = random.choice(MALIGNANT_PROMPTS)
        
        # Tokenize caption
        try:
            text_inputs = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            caption_ids = text_inputs['input_ids'][0]
            attention_mask = text_inputs['attention_mask'][0]
        except TypeError:
            tokenized = self.tokenizer([caption])
            caption_ids = tokenized[0] if tokenized.dim() == 2 else tokenized
            attention_mask = torch.ones_like(caption_ids)
        
        return {
            'imgs': pixel_values,
            'caption_ids': caption_ids,
            'attention_mask': attention_mask,
            'token_type_ids': torch.zeros_like(caption_ids),
            'label': 0 if sample['label'] == 'benign' else 1,
            'image_path': str(sample['image_path'])
        }
