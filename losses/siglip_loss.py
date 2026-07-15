#!/usr/bin/env python3
"""
SigLIP Loss (Sigmoid Loss for Language-Image Pre-training)
===========================================================
From "Sigmoid Loss for Language Image Pre-Training" (Google Research)
Paper: https://arxiv.org/abs/2303.15343

Uses pairwise sigmoid loss instead of softmax-based InfoNCE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    """
    SigLIP Loss using sigmoid-based pairwise loss.
    
    Args:
        temperature (float): Temperature parameter to scale similarities
        bias (float): Initial bias value (learnable parameter)
    """
    def __init__(self, temperature=1.0, bias=0.0):
        super(SigLIPLoss, self).__init__()
        self.temperature = temperature
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, image_features, text_features):
        """
        Compute SigLIP loss.
        
        Args:
            image_features: Image embeddings [batch_size, embed_dim]
            text_features: Text embeddings [batch_size, embed_dim]
            
        Returns:
            loss: Scalar loss value
        """
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute pairwise similarities
        logits = torch.matmul(image_features, text_features.t()) / self.temperature

        # Create labels matrix (1 for positive pairs, -1 for negatives)
        batch_size = image_features.shape[0]
        labels = 2 * torch.eye(batch_size, device=image_features.device) - 1

        # Sigmoid loss: -log(sigmoid(z * t)) where z is logit, t is label
        loss = -F.logsigmoid(labels * (logits + self.bias)).sum() / batch_size

        return loss
