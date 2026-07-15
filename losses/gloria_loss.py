#!/usr/bin/env python3
"""
GLoRIA Loss (Global-Local Representations for Images)
======================================================
Combines global and local contrastive losses with attention mechanism.

From "GLoRIA: A Multimodal Global-Local Representation Learning Framework 
for Label-efficient Medical Image Recognition"
Paper: https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):
    """
    Compute attention between query (text) and context (image).
    
    Args:
        query: [batch, ndf, queryL] - text features
        context: [batch, ndf, ih, iw] - image features
        temp1: temperature for attention
    
    Returns:
        weightedContext: [batch, ndf, queryL] - attended image features
        attn: [batch, queryL, ih, iw] - attention maps
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # Reshape context: [batch, ndf, sourceL]
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()  # [batch, sourceL, ndf]

    # Compute attention: [batch, sourceL, ndf] x [batch, ndf, queryL] -> [batch, sourceL, queryL]
    attn = torch.bmm(contextT, query)
    attn = attn.view(batch_size * sourceL, queryL)
    attn = F.softmax(attn, dim=-1)

    # Reshape and apply temperature
    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()  # [batch, queryL, sourceL]
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = F.softmax(attn, dim=-1)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()  # [batch, sourceL, queryL]

    # Apply attention to context
    weightedContext = torch.bmm(context, attnT)  # [batch, ndf, queryL]

    return weightedContext, attn.view(batch_size, queryL, ih, iw)


class GLoRIALoss(nn.Module):
    """
    GLoRIA Loss combining global and local contrastive learning.
    
    Args:
        temp_global: Temperature for global contrastive loss
        temp_local1: Temperature for local attention
        temp_local2: Temperature for local similarity
        temp_local3: Temperature for local contrastive loss
        lambda_global: Weight for global loss
        lambda_local: Weight for local loss
        local_agg: Aggregation method for local similarities ('sum' or 'mean')
    """
    def __init__(
        self,
        temp_global=10.0,
        temp_local1=4.0,
        temp_local2=5.0,
        temp_local3=10.0,
        lambda_global=1.0,
        lambda_local=1.0,
        local_agg="sum"
    ):
        super().__init__()
        self.temp_global = temp_global
        self.temp_local1 = temp_local1
        self.temp_local2 = temp_local2
        self.temp_local3 = temp_local3
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.local_agg = local_agg
        self.loss_fn = nn.CrossEntropyLoss()
    
    def global_loss(self, img_features, text_features, eps=1e-8):
        """
        Compute global contrastive loss.
        
        Args:
            img_features: [batch, dim] - global image features
            text_features: [batch, dim] - global text features
        
        Returns:
            loss_i2t: image-to-text loss
            loss_t2i: text-to-image loss
        """
        batch_size = img_features.shape[0]
        labels = torch.arange(batch_size, device=img_features.device)

        # Normalize features
        img_features_norm = F.normalize(img_features, p=2, dim=1)
        text_features_norm = F.normalize(text_features, p=2, dim=1)

        # Compute similarity matrix
        scores = torch.mm(img_features_norm, text_features_norm.t()) * self.temp_global
        
        # Compute losses
        loss_i2t = self.loss_fn(scores, labels)
        loss_t2i = self.loss_fn(scores.t(), labels)
        
        return loss_i2t, loss_t2i
    
    def local_loss(self, img_features_local, text_features_local, cap_lens):
        """
        Compute local contrastive loss with attention.
        
        Args:
            img_features_local: [batch, dim, h, w] - local image features
            text_features_local: [batch, dim, seq_len] - local text features
            cap_lens: [batch] - actual text lengths
        
        Returns:
            loss_i2t: image-to-text local loss
            loss_t2i: text-to-image local loss
            att_maps: attention maps
        """
        batch_size = img_features_local.shape[0]
        
        att_maps = []
        similarities = []
        
        for i in range(batch_size):
            # Get the i-th text description
            words_num = cap_lens[i]
            word = text_features_local[i, :, :words_num].unsqueeze(0).contiguous()
            word = word.repeat(batch_size, 1, 1)  # [batch, dim, words_num]
            context = img_features_local  # [batch, dim, h, w]
            
            # Compute attention
            weiContext, attn = attention_fn(word, context, self.temp_local1)
            
            att_maps.append(attn[i].unsqueeze(0).contiguous())
            
            # Transpose for similarity computation
            word = word.transpose(1, 2).contiguous()  # [batch, words_num, dim]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [batch, words_num, dim]
            
            # Flatten
            word = word.reshape(batch_size * words_num, -1)
            weiContext = weiContext.reshape(batch_size * words_num, -1)
            
            # Compute cosine similarity
            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)
            
            # Aggregate similarities
            row_sim = row_sim * self.temp_local2
            row_sim = torch.exp(row_sim)
            if self.local_agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)
            
            similarities.append(row_sim)
        
        # Stack similarities
        similarities = torch.cat(similarities, 1)  # [batch, batch]
        similarities = similarities * self.temp_local3
        similarities_t = similarities.transpose(0, 1)
        
        labels = torch.arange(batch_size, device=similarities.device)
        
        loss_i2t = self.loss_fn(similarities, labels)
        loss_t2i = self.loss_fn(similarities_t, labels)
        
        return loss_i2t, loss_t2i, att_maps
    
    def forward(self, image_features_dict, text_features_dict):
        """
        Compute GLoRIA loss.
        
        Args:
            image_features_dict: Dict with keys:
                - 'global': [B, D] global image features
                - 'local': [B, D, H, W] local image features (CNN feature maps)
            text_features_dict: Dict with keys:
                - 'global': [B, D] global text features
                - 'local': [B, D, L] local text features (word embeddings)
                - 'attention_mask': [B, L] attention mask
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary with individual losses
        """
        # Extract features
        img_global = image_features_dict['global']
        img_local = image_features_dict['local']
        text_global = text_features_dict['global']
        text_local = text_features_dict['local']
        attention_mask = text_features_dict['attention_mask']
        
        # Compute caption lengths (number of non-padding tokens)
        cap_lens = attention_mask.sum(dim=1).long()
        
        # Global loss
        loss_global_i2t, loss_global_t2i = self.global_loss(img_global, text_global)
        loss_global = (loss_global_i2t + loss_global_t2i) / 2.0
        
        # Local loss
        # Reshape local features if needed
        # Text: [B, L, D] -> [B, D, L]
        if text_local.dim() == 3 and text_local.size(1) != text_local.size(2):
            text_local = text_local.transpose(1, 2)
        
        # Image: [B, N, D] -> [B, D, H, W]
        # Assume square feature map
        if img_local.dim() == 3:
            B, N, D = img_local.shape
            H = W = int(N ** 0.5)
            img_local = img_local.transpose(1, 2).reshape(B, D, H, W)
        
        loss_local_i2t, loss_local_t2i, att_maps = self.local_loss(
            img_local, text_local, cap_lens
        )
        loss_local = (loss_local_i2t + loss_local_t2i) / 2.0
        
        # Total loss
        total_loss = (
            self.lambda_global * loss_global +
            self.lambda_local * loss_local
        )
        
        return total_loss, {
            'loss_global': loss_global.item(),
            'loss_local': loss_local.item(),
        }
