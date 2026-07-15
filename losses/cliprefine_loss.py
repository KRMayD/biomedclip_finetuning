#!/usr/bin/env python3
"""
CLIP-Refine Loss with Random Feature Regularization
====================================================
From "CLIP-Refine: Learning Robust Vision-Language Models from Noisy Captions"

Combines standard CLIP contrastive loss with random feature regularization.
The random feature regularization prevents overfitting by pushing features 
towards a random distribution, which helps maintain the model's generalization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists


class CLIPRefineLoss(nn.Module):
    """
    CLIP-Refine Loss: CLIP contrastive loss + Random feature regularization.
    
    The loss has two components:
    1. Standard CLIP contrastive loss (symmetric cross-entropy)
    2. MSE loss between features and random features sampled from a distribution
    
    Args:
        temperature (float): Temperature for contrastive loss (default: 1.0)
        lambda_rand (float): Weight for random feature regularization (default: 1.0)
        strategy (str): Random sampling strategy. Options:
            - 'std_sample': Sample from Normal(mu, sigma)
            - 'uniform_sample': Sample from Uniform(mu, sigma)
            - 'precomputed_sample': Sample from precomputed statistics
            - 'precomputed_fixed': Use fixed precomputed mean
            - 'uniform_fixed': Use fixed uniform value
        share_random_feat (bool): Whether to share same random feature for image and text (default: True)
        mu (float): Mean for random distribution (default: 0.0)
        sigma (float): Standard deviation or upper bound for random distribution (default: 1.0)
        precomputed_stats (str): Path to .npz file with precomputed statistics (optional)
        regularization_decay (bool): Whether to decay regularization weight over training (default: False)
        max_iteration (int): Maximum iterations for decay schedule (required if regularization_decay=True)
    """
    def __init__(
        self,
        temperature=1.0,
        lambda_rand=1.0,
        strategy="std_sample",
        share_random_feat=True,
        mu=0.0,
        sigma=1.0,
        precomputed_stats=None,
        regularization_decay=False,
        max_iteration=None,
        normalize_features=True,
    ):
        super(CLIPRefineLoss, self).__init__()
        self.temperature = temperature
        self.lambda_rand = lambda_rand
        self.strategy = strategy
        self.share_random_feat = share_random_feat
        self.regularization_decay = regularization_decay
        self.max_iteration = max_iteration
        self.normalize_features = normalize_features
        self.current_iteration = 0
        self.decay_rate = 1.0
        
        # Setup random distribution based on strategy
        if self.strategy == "std_sample":
            self.register_buffer('mean', torch.tensor([mu]))
            self.register_buffer('std', torch.tensor([sigma]))
            self.random_dist = dists.Normal(loc=torch.tensor([mu]), scale=torch.tensor([sigma]))
        elif self.strategy in ["uniform_sample", "uniform_fixed"]:
            self.register_buffer('mean', torch.tensor([mu]))
            self.register_buffer('std', torch.tensor([sigma]))
            self.random_dist = dists.Uniform(low=torch.tensor([mu]), high=torch.tensor([sigma]))
        elif self.strategy in ["precomputed_fixed", "precomputed_sample"]:
            assert precomputed_stats is not None, "precomputed_stats path required for precomputed strategies"
            stats = np.load(precomputed_stats)
            self.register_buffer('mean', torch.from_numpy(stats["mean"]))
            self.register_buffer('std', torch.from_numpy(stats["std"]))
            if self.strategy == "precomputed_sample":
                self.random_dist = dists.Normal(loc=self.mean, scale=self.std)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from: std_sample, uniform_sample, "
                           f"uniform_fixed, precomputed_sample, precomputed_fixed")
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.feature_loss_fn = F.mse_loss

    def generate_random_feature(self, size):
        """Generate random features based on the selected strategy."""
        if self.strategy == "std_sample":
            f_rand = self.random_dist.sample(size).to(self.mean.device)
        elif self.strategy == "uniform_sample":
            f_rand = self.random_dist.sample(size).to(self.mean.device)
        elif self.strategy == "precomputed_sample":
            f_rand = self.random_dist.sample([size[0]]).to(self.mean.device)
        elif self.strategy in ["precomputed_fixed", "uniform_fixed"]:
            # Expand fixed value to match the target size
            f_rand = self.mean.expand(size)
        else:
            raise NotImplementedError
        return f_rand.squeeze()

    def update_decay_rate(self):
        """Update decay rate for regularization weight."""
        if self.regularization_decay:
            assert self.max_iteration is not None, "max_iteration required when regularization_decay=True"
            assert self.current_iteration <= self.max_iteration
            self.decay_rate = 1.0 - (self.current_iteration / self.max_iteration)

    def clip_contrastive_loss(self, image_features, text_features):
        """Compute standard CLIP contrastive loss."""
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

    def random_feature_regularization(self, image_features, text_features):
        """Compute random feature regularization loss."""
        # Generate random features
        if self.share_random_feat:
            # Use same random feature for both modalities
            feat_rand = self.generate_random_feature(image_features.size())
            loss_feat = self.feature_loss_fn(image_features, feat_rand) + \
                       self.feature_loss_fn(text_features, feat_rand)
        else:
            # Use different random features for each modality
            feat_rand_img = self.generate_random_feature(image_features.size())
            feat_rand_txt = self.generate_random_feature(text_features.size())
            loss_feat = self.feature_loss_fn(image_features, feat_rand_img) + \
                       self.feature_loss_fn(text_features, feat_rand_txt)
        
        return loss_feat

    def forward(self, image_features, text_features):
        """
        Compute CLIP-Refine loss.
        
        Args:
            image_features: Image embeddings [batch_size, embed_dim]
            text_features: Text embeddings [batch_size, embed_dim]
            
        Returns:
            loss: Scalar loss value
        """
        # Update decay rate if using regularization decay
        self.update_decay_rate()
        
        # Compute contrastive loss
        contrastive_loss = self.clip_contrastive_loss(image_features, text_features)
        
        # Compute random feature regularization
        reg_loss = self.random_feature_regularization(image_features, text_features)
        
        # Combine losses with decay rate
        total_loss = contrastive_loss + self.decay_rate * self.lambda_rand * reg_loss
        
        # Increment iteration counter
        self.current_iteration += 1
        
        return total_loss

    def get_loss_components(self, image_features, text_features):
        """
        Get individual loss components for logging.
        
        Returns:
            dict: Dictionary with 'contrastive', 'regularization', and 'total' losses
        """
        self.update_decay_rate()
        
        contrastive_loss = self.clip_contrastive_loss(image_features, text_features)
        reg_loss = self.random_feature_regularization(image_features, text_features)
        total_loss = contrastive_loss + self.decay_rate * self.lambda_rand * reg_loss
        
        self.current_iteration += 1
        
        return {
            'contrastive': contrastive_loss.item(),
            'regularization': reg_loss.item(),
            'total': total_loss.item(),
            'decay_rate': self.decay_rate,
        }
