#!/usr/bin/env python3
"""
Hard Negative Loss (HN-NCE)
============================
Hard Negative Noise Contrastive Estimation
From "Hard Negative Noise Contrastive Estimation" 
Paper: https://arxiv.org/abs/2301.02280

Reweights hard negatives to improve contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeLoss(nn.Module):
    """
    Hard Negative Loss with reweighting of hard negatives.
    
    Args:
        temperature (float): Temperature to control sharpness of distribution
        beta1 (float): Hardness parameter for image features (reweighting strength)
        beta2 (float): Hardness parameter for text features
        alpha (float): Weighting of positive samples (0 = decoupled DHN-NCE)
    """
    def __init__(self, temperature=1.0, beta1=1.0, beta2=1.0, alpha=0.0):
        super(HardNegativeLoss, self).__init__()
        self.temperature = temperature
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha

    def forward(self, image_features, text_features):
        """
        Compute Hard Negative Loss.
        
        Args:
            image_features: Image embeddings [batch_size, embed_dim]
            text_features: Text embeddings [batch_size, embed_dim]
            
        Returns:
            loss: Scalar loss value
        """
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        batch_size = logits_per_image.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool, device=image_features.device)
        neg_mask = ~mask

        # Positive pairs: diagonal elements
        pos = torch.exp(logits_per_image * mask)

        # Hard negative reweighting
        N = batch_size - 1
        
        # Image-to-text hard negatives
        norm_term_img = torch.sum(torch.exp(logits_per_image * neg_mask), dim=-1, keepdim=True)
        reweight_img = N * (torch.exp(self.beta1 * logits_per_image * neg_mask)) / (norm_term_img + 1e-8)
        neg_img = reweight_img * torch.exp(logits_per_image * neg_mask)
        
        # Text-to-image hard negatives
        norm_term_text = torch.sum(torch.exp(logits_per_text * neg_mask), dim=-1, keepdim=True)
        reweight_text = N * (torch.exp(self.beta2 * logits_per_text * neg_mask)) / (norm_term_text + 1e-8)
        neg_text = reweight_text * torch.exp(logits_per_text * neg_mask)

        # Calculate loss
        pos_diag = torch.diag(pos)
        loss_img = -torch.log(pos_diag / (pos_diag * self.alpha + neg_img.sum(dim=-1) + 1e-8))
        loss_text = -torch.log(pos_diag / (pos_diag * self.alpha + neg_text.sum(dim=-1) + 1e-8))
        
        loss = (loss_img + loss_text).mean()

        return loss
