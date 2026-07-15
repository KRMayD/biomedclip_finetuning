#!/usr/bin/env python3
"""
BiomedCLIP Model Wrapper
========================
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor


def load_biomedclip(checkpoint="chuhac/BiomedCLIP-vit-bert-hf", device="cuda"):
    """
    Load BiomedCLIP model, tokenizer, and processor.
    
    Args:
        checkpoint (str): HuggingFace checkpoint name
        device (str): Device to load model on
        
    Returns:
        model: BiomedCLIP model
        tokenizer: Text tokenizer
        processor: Image processor
    """
    print(f"[INFO] Loading BiomedCLIP from {checkpoint}...")

    if checkpoint.startswith("hf-hub:"):
        import sys
        open_clip_src = "/home/msko021220/finegrained-vlm-training/biomedclip_finetuning/open_clip/src"
        if open_clip_src not in sys.path:
            sys.path.insert(0, open_clip_src)
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            checkpoint,
            pretrained=None,
            device=device,
            precision="fp32",
        )
        tokenizer = open_clip.get_tokenizer(checkpoint)
        model.to(device)
        print(f"[INFO] OpenCLIP BiomedCLIP loaded successfully on {device}")
        return model, tokenizer, preprocess

    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    # Get image processor
    if hasattr(processor, 'image_processor'):
        image_processor = processor.image_processor
    else:
        image_processor = processor

    model.to(device)
    print(f"[INFO] HuggingFace BiomedCLIP loaded successfully on {device}")

    return model, tokenizer, image_processor


def _openclip_state_dict(checkpoint_path):
    """Load a plain or wrapped OpenCLIP state dict from a trusted local checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"OpenCLIP checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    # Older BiomedCLIP checkpoints persist this deterministic BERT buffer, while
    # the current OpenCLIP text encoder recreates it internally.
    state_dict.pop("text.transformer.embeddings.position_ids", None)
    return state_dict


def load_openclip(
    checkpoint,
    model_name="ViT-B-32-quickgelu",
    device="cuda",
    openclip_src="/home/msko021220/finegrained-vlm-training/biomedclip_finetuning/open_clip/src",
):
    """Load an OpenCLIP model from a local state dict for CLIP-style post-training."""
    openclip_src = Path(openclip_src)
    if not openclip_src.is_dir():
        raise FileNotFoundError(f"OpenCLIP source directory does not exist: {openclip_src}")
    if str(openclip_src) not in sys.path:
        sys.path.insert(0, str(openclip_src))

    import open_clip

    print(f"[INFO] Loading OpenCLIP {model_name} from {checkpoint}...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
    missing, unexpected = model.load_state_dict(_openclip_state_dict(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "OpenCLIP checkpoint/model mismatch: "
            f"missing={missing[:10]} ({len(missing)} total), "
            f"unexpected={unexpected[:10]} ({len(unexpected)} total)"
        )

    model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    print(f"[INFO] OpenCLIP loaded successfully on {device}")
    return model, tokenizer, preprocess


def load_model(
    backend="biomedclip",
    checkpoint="chuhac/BiomedCLIP-vit-bert-hf",
    device="cuda",
    openclip_model="ViT-B-32-quickgelu",
    openclip_src="/home/msko021220/finegrained-vlm-training/biomedclip_finetuning/open_clip/src",
):
    """Load the selected vision-language backend while keeping legacy BiomedCLIP behavior."""
    if backend == "biomedclip":
        return load_biomedclip(checkpoint, device)
    if backend == "openclip":
        return load_openclip(checkpoint, openclip_model, device, openclip_src)
    raise ValueError(f"Unsupported model backend: {backend}")


def get_biomedclip_features(model, batch):
    """
    Extract image and text features from BiomedCLIP.
    
    Args:
        model: BiomedCLIP model
        batch: Dictionary containing 'imgs', 'caption_ids', 'attention_mask'
        
    Returns:
        image_features: Image embeddings [batch_size, embed_dim]
        text_features: Text embeddings [batch_size, embed_dim]
    """
    if hasattr(model, "encode_image") and hasattr(model, "encode_text"):
        image_features = model.encode_image(batch['imgs'])
        text_features = model.encode_text(batch['caption_ids'])
        return image_features, text_features

    # Image features
    image_outputs = model.vision_model(batch['imgs'])
    if isinstance(image_outputs, tuple):
        hidden_states = image_outputs[0]
        image_features = hidden_states[:, 0]  # CLS token
    elif hasattr(image_outputs, 'pooler_output'):
        image_features = image_outputs.pooler_output
    else:
        image_features = image_outputs.last_hidden_state[:, 0]
    
    # Text features
    text_outputs = model.text_model(
        input_ids=batch['caption_ids'],
        attention_mask=batch['attention_mask']
    )
    if isinstance(text_outputs, tuple):
        hidden_states = text_outputs[0]
        text_features = hidden_states[:, 0]  # CLS token
    elif hasattr(text_outputs, 'pooler_output'):
        text_features = text_outputs.pooler_output
    else:
        text_features = text_outputs.last_hidden_state[:, 0]
    
    return image_features, text_features


def get_biomedclip_features_mgca(model, batch):
    """
    Extract global and local features from BiomedCLIP for MGCA loss.
    
    Args:
        model: BiomedCLIP model
        batch: Dictionary containing 'imgs', 'caption_ids', 'attention_mask'
        
    Returns:
        image_features_dict: Dict with 'global' [B, D], 'local' [B, N, D], 'attention_map' [B, num_heads, N+1, N+1]
        text_features_dict: Dict with 'global' [B, D], 'local' [B, L, D], 'attention_mask' [B, L], 'attention_weights' [B, L-1]
    """
    # Image features (global + local)
    image_outputs = model.vision_model(batch['imgs'], output_hidden_states=True, output_attentions=True)
    
    if hasattr(image_outputs, 'last_hidden_state'):
        hidden_states = image_outputs.last_hidden_state  # [B, 1+N, D]
    else:
        hidden_states = image_outputs[0]
    
    img_global = hidden_states[:, 0]  # CLS token [B, D]
    img_local = hidden_states[:, 1:]  # Patch tokens [B, N, D]
    
    # Extract attention map from last layer
    img_attn_map = None
    if hasattr(image_outputs, 'attentions') and image_outputs.attentions is not None:
        img_attn_map = image_outputs.attentions[-1]  # Last layer attention [B, num_heads, N+1, N+1]
    
    # Text features (global + local) with attention extraction via hook
    # BiomedCLIP's text_model doesn't support output_attentions in the standard way,
    # so we use a forward hook to capture attention from the last layer
    captured_attention = [None]
    
    def attention_hook(module, input, output):
        """Capture attention weights from self_attn module"""
        if isinstance(output, tuple) and len(output) > 1:
            # output is typically (hidden_states, attention_weights)
            attn_weights = output[1]
            if torch.is_tensor(attn_weights) and len(attn_weights.shape) == 4:
                captured_attention[0] = attn_weights.detach()
    
    # Register hook on last layer's self-attention
    last_layer = model.text_model.encoder.layers[-1]
    hook = last_layer.self_attn.register_forward_hook(attention_hook)
    
    # Forward pass
    text_outputs = model.text_model(
        input_ids=batch['caption_ids'],
        attention_mask=batch['attention_mask'],
        output_hidden_states=True
    )
    
    # Remove hook
    hook.remove()
    
    # Extract hidden states
    if hasattr(text_outputs, 'last_hidden_state'):
        text_hidden = text_outputs.last_hidden_state  # [B, L, D]
    elif isinstance(text_outputs, tuple):
        text_hidden = text_outputs[0]  # [B, L, D]
    else:
        text_hidden = text_outputs
    
    text_global = text_hidden[:, 0]  # CLS token [B, D]
    text_local = text_hidden  # All tokens [B, L, D]
    
    # Extract BERT attention weights from captured attention
    # Same as MGCA: last_layer_attn[:, :, 0, 1:].mean(dim=1)
    text_attn_weights = None
    if captured_attention[0] is not None:
        # captured_attention[0] shape: [B, num_heads, L, L]
        # Get CLS token's attention to other tokens (excluding CLS itself)
        last_layer_attn = captured_attention[0]
        text_attn_weights = last_layer_attn[:, :, 0, 1:].mean(dim=1)  # [B, L-1]
    
    return {
        'global': img_global,
        'local': img_local,
        'attention_map': img_attn_map
    }, {
        'global': text_global,
        'local': text_local,
        'attention_mask': batch['attention_mask'],
        'attention_weights': text_attn_weights  # BERT attention weights
    }
