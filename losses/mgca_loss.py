#!/usr/bin/env python3
"""
MGCA Loss (Multi-Granularity Cross-Modal Alignment)
===================================================
Combines global, local (token-level), and prototype-level alignment.

From MGCA paper - requires additional projection heads and prototype layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GlobalEmbedding(nn.Module):
    """Project global features to embedding space"""
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    """Project local features to embedding space"""
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        # x: [B, N, D] -> [B, D, N]
        x = x.permute(0, 2, 1)
        x = self.head(x)
        # [B, D, N] -> [B, N, D]
        return x.permute(0, 2, 1)


class MGCALoss(nn.Module):
    """
    MGCA Loss with multi-granularity alignment.
    
    Combines:
    - Global image-text alignment (ITA)
    - Local token-level alignment
    - Prototype-based alignment
    
    Args:
        emb_dim: Embedding dimension
        hidden_dim: Hidden dimension for projection heads
        num_prototypes: Number of prototypes
        softmax_temperature: Temperature for global alignment
        local_temperature: Temperature for local alignment
        proto_temperature: Temperature for prototype alignment
        lambda_1: Weight for global loss
        lambda_2: Weight for local loss
        lambda_3: Weight for prototype loss
        epsilon: Sinkhorn-Knopp epsilon
        sinkhorn_iterations: Number of Sinkhorn iterations
        bidirectional: Use bidirectional local alignment
        use_local_atten: Use attention for local alignment
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        input_dim=768,
        emb_dim=128,
        hidden_dim=2048,
        num_prototypes=500,
        softmax_temperature=0.07,
        local_temperature=0.1,
        proto_temperature=0.2,
        lambda_1=1.0,
        lambda_2=1.0,
        lambda_3=1.0,
        epsilon=0.05,
        sinkhorn_iterations=3,
        bidirectional=True,
        use_local_atten=False,
        num_heads=1
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.softmax_temperature = softmax_temperature
        self.local_temperature = local_temperature
        self.proto_temperature = proto_temperature
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.bidirectional = bidirectional
        self.use_local_atten = use_local_atten
        
        # Projection heads for images
        self.img_global_embed = GlobalEmbedding(input_dim, hidden_dim, emb_dim)
        self.img_local_embed = LocalEmbedding(input_dim, hidden_dim, emb_dim)
        
        # Projection heads for text
        self.text_global_embed = GlobalEmbedding(input_dim, hidden_dim, emb_dim)
        self.text_local_embed = LocalEmbedding(input_dim, hidden_dim, emb_dim)
        
        # Attention layers (optional)
        self.patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        
        # Prototype layer
        self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)
    
    def sinkhorn(self, Q, nmb_iters):
        """Sinkhorn-Knopp algorithm for optimal transport"""
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q
            
            K, B = Q.shape
            
            u = torch.zeros(K).to(Q.device)
            r = torch.ones(K).to(Q.device) / K
            c = torch.ones(B).to(Q.device) / B
            
            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / (u + 1e-8)).unsqueeze(1)
                Q *= (c / (torch.sum(Q, dim=0) + 1e-8)).unsqueeze(0)
            
            return (Q / (torch.sum(Q, dim=0, keepdim=True) + 1e-8)).t().float()
    
    def forward(self, image_features_dict, text_features_dict):
        """
        Compute MGCA loss.
        
        Args:
            image_features_dict: Dict with keys:
                - 'global': [B, D] global image features (CLS token)
                - 'local': [B, N, D] local image features (patch tokens)
            text_features_dict: Dict with keys:
                - 'global': [B, D] global text features (CLS token)
                - 'local': [B, L, D] local text features (word tokens)
                - 'attention_mask': [B, L] attention mask
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary with individual losses
        """
        # Extract features
        img_feat = image_features_dict['global']  # [B, D]
        patch_feat = image_features_dict['local']  # [B, N, D]
        img_attn_map = image_features_dict.get('attention_map', None)  # [B, num_heads, N+1, N+1] or None
        report_feat = text_features_dict['global']  # [B, D]
        word_feat = text_features_dict['local']  # [B, L, D]
        
        # Use BERT attention weights if available, otherwise fall back to attention mask
        if text_features_dict.get('attention_weights') is not None:
            # BERT attention weights (semantic importance) - same as MGCA
            word_attn = text_features_dict['attention_weights']  # [B, L-1]
            # Pad to match word_feat length (add CLS token attention)
            padding = torch.ones(word_attn.size(0), 1).to(word_attn.device) * word_attn.mean()
            word_attn = torch.cat([padding, word_attn], dim=1)  # [B, L]
        else:
            # Fallback: use attention mask (0/1 padding indicator)
            word_attn = text_features_dict['attention_mask'].float()  # [B, L]
        
        # Project to embedding space
        img_emb = self.img_global_embed(img_feat)
        img_emb = F.normalize(img_emb, dim=-1)
        
        patch_emb = self.img_local_embed(patch_feat)
        patch_emb = F.normalize(patch_emb, dim=-1)
        
        report_emb = self.text_global_embed(report_feat)
        report_emb = F.normalize(report_emb, dim=-1)
        
        word_emb = self.text_local_embed(word_feat)
        word_emb = F.normalize(word_emb, dim=-1)
        
        bz = img_emb.size(0)
        labels = torch.arange(bz).type_as(report_emb).long()
        
        # ============ Global Image-Text Alignment ============
        scores = img_emb.mm(report_emb.t())
        scores /= self.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_ita = loss0 + loss1
        
        # ============ Local Token-level Alignment ============
        mask = ~word_attn.bool()  # True = padding
        
        # Text-to-Image (word to patch)
        if self.use_local_atten:
            word_atten_output, _ = self.word_local_atten_layer(
                word_emb, patch_emb, patch_emb)
        else:
            atten_sim = torch.bmm(word_emb, patch_emb.permute(0, 2, 1))
            word_num = word_emb.size(1)
            atten_scores = F.softmax(atten_sim / self.local_temperature, dim=-1)
            word_atten_output = torch.bmm(atten_scores, patch_emb)
        
        word_atten_output = F.normalize(word_atten_output, dim=-1)
        
        # Compute word attention weights
        with torch.no_grad():
            word_atten_weights = []
            for i in range(bz):
                atten_weight = word_attn[i].clone()
                nonzero = atten_weight.nonzero().squeeze()
                if nonzero.numel() > 0:
                    low = torch.quantile(atten_weight[nonzero], 0.1)
                    high = torch.quantile(atten_weight[nonzero], 0.9)
                    atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight)
            word_atten_weights = torch.stack(word_atten_weights)
            word_atten_weights /= (word_atten_weights.sum(dim=1, keepdims=True) + 1e-8)
        
        word_sim = torch.bmm(word_emb, word_atten_output.permute(0, 2, 1)) / self.local_temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(word_emb).long().repeat(bz)
        loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz
        
        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
            word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz
        
        loss_word = (loss_word_1 + loss_word_2) / 2.
        
        if self.bidirectional:
            # Image-to-Text (patch to word)
            if self.use_local_atten:
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb, word_emb, word_emb, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb, word_emb.permute(0, 2, 1))
                patch_num = patch_emb.size(1)
                atten_sim[mask.unsqueeze(1).repeat(1, patch_num, 1)] = float("-inf")
                atten_scores = F.softmax(atten_sim / self.local_temperature, dim=-1)
                patch_atten_output = torch.bmm(atten_scores, word_emb)
            
            patch_atten_output = F.normalize(patch_atten_output, dim=-1)
            
            # Compute patch attention weights from ViT attention map
            patch_num = patch_emb.size(1)
            if img_attn_map is not None:
                # Extract CLS token attention to patches from last layer
                # img_attn_map: [B, num_heads, N+1, N+1]
                # Take attention from CLS (idx 0) to patches (idx 1:)
                with torch.no_grad():
                    atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)  # [B, N] average over heads
                    patch_atten_weights = []
                    for i in range(bz):
                        atten_weight = atten_weights[i]
                        # Clip to 10-90 percentile
                        low = torch.quantile(atten_weight, 0.1)
                        high = torch.quantile(atten_weight, 0.9)
                        atten_weight = atten_weight.clip(low, high)
                        patch_atten_weights.append(atten_weight)
                    patch_atten_weights = torch.stack(patch_atten_weights)
                    patch_atten_weights /= (patch_atten_weights.sum(dim=1, keepdim=True) + 1e-8)
            else:
                # Fallback to uniform weights if attention map not available
                patch_atten_weights = torch.ones(bz, patch_num).to(patch_emb.device) / patch_num
            
            patch_sim = torch.bmm(patch_emb, patch_atten_output.permute(0, 2, 1)) / self.local_temperature
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(patch_emb).long().repeat(bz)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz
            
            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz
            
            loss_patch = (loss_patch_1 + loss_patch_2) / 2.
            loss_local = loss_patch + loss_word
        else:
            loss_local = loss_word
        
        # ============ Prototype-based Alignment ============
        # Normalize prototype layer
        with torch.no_grad():
            w = self.prototype_layer.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototype_layer.weight.copy_(w)
        
        # Compute prototype scores
        img_proto_out = self.prototype_layer(img_emb)
        report_proto_out = self.prototype_layer(report_emb)
        
        # Compute assignment codes using Sinkhorn-Knopp
        with torch.no_grad():
            img_code = torch.exp(img_proto_out / self.epsilon).t()
            img_code = self.sinkhorn(img_code, self.sinkhorn_iterations)
            
            report_code = torch.exp(report_proto_out / self.epsilon).t()
            report_code = self.sinkhorn(report_code, self.sinkhorn_iterations)
        
        img_proto_prob = F.softmax(img_proto_out / self.proto_temperature, dim=1)
        report_proto_prob = F.softmax(report_proto_out / self.proto_temperature, dim=1)
        
        loss_i2t_proto = -torch.mean(
            torch.sum(img_code * torch.log(report_proto_prob + 1e-8), dim=1))
        loss_t2i_proto = -torch.mean(
            torch.sum(report_code * torch.log(img_proto_prob + 1e-8), dim=1))
        
        loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.
        
        # Total loss
        total_loss = (
            self.lambda_1 * loss_ita +
            self.lambda_2 * loss_local +
            self.lambda_3 * loss_proto
        )
        
        return total_loss, {
            'loss_global': loss_ita.item(),
            'loss_local': loss_local.item(),
            'loss_proto': loss_proto.item(),
        }
