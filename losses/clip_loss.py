#!/usr/bin/env python3
"""
Standard CLIP Contrastive Loss (InfoNCE)
=========================================
From "Learning Transferable Visual Models From Natural Language Supervision"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    Standard CLIP Contrastive Loss using InfoNCE objective.
    
    Args:
        temperature (float): Temperature parameter to scale logits
    """
    def __init__(self, temperature=1.0, normalize_features=True):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.normalize_features = normalize_features
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        """
        Compute CLIP loss.
        
        Args:
            image_features: Image embeddings [batch_size, embed_dim]
            text_features: Text embeddings [batch_size, embed_dim]
            
        Returns:
            loss: Scalar loss value
        """
        if self.normalize_features:
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)

        # Compute logits
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        # Labels: diagonal elements are correct pairs
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Compute symmetric loss
        loss_i = self.loss_fn(logits_per_image, labels)
        loss_t = self.loss_fn(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return loss
