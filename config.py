#!/usr/bin/env python3
"""
Configuration file for BiomedCLIP fine-tuning
=============================================
Contains prompts, default hyperparameters, and common settings.
"""

# ==================== Text Prompts ====================
BENIGN_PROMPTS = [
    "A medical breast mammogram showing a well-defined, round mass suggestive of a benign breast tumor.",
    "A medical breast mammogram showing a smooth, well-circumscribed mass suggestive of a benign breast tumor.",
    "A mammogram displaying a round, homogeneous mass indicative of a benign breast tumor.",
    "A breast imaging study revealing a well-defined, encapsulated mass likely representing a benign breast tumor.",
    "A mammogram showing a sharply marginated, round mass suggestive of a benign breast tumor.",
    "A breast scan identifying a well-defined, oval mass suggestive of a benign breast tumor.",
    "A medical breast mammogram displaying a lobulated, non-aggressive mass consistent with a benign tumor.",
    "A breast imaging scan showing a well-circumscribed, smooth mass indicative of a benign breast tumor.",
    "A mammogram revealing a low-density, well-delineated mass likely representing a benign breast tumor.",
    "A breast imaging study showing a well-margined, round mass suggestive of a benign breast tumor.",
    "A medical mammogram detecting a smooth, well-defined mass characteristic of a benign breast tumor.",
    "A mammogram displaying a non-invasive, well-defined mass suggestive of a benign breast tumor.",
    "A breast scan showing a well-contained, round mass likely indicating a benign breast tumor.",
    "A breast mammogram identifying a well-encapsulated, smooth mass consistent with a benign tumor.",
    "A medical mammogram revealing a well-defined, non-irregular mass suggestive of a benign breast tumor.",
    "A breast imaging study showing a homogenous, well-circumscribed mass indicative of a benign tumor.",
    "A mammogram displaying a non-spiculated, round mass suggestive of a benign breast tumor.",
    "A breast scan showing a smooth, encapsulated mass likely representing a benign breast tumor.",
    "A medical breast mammogram detecting a non-aggressive, well-defined mass characteristic of a benign tumor.",
    "A breast imaging study revealing a well-marginated, round mass indicative of a benign breast tumor.",
    "A mammogram displaying a smoothly contoured mass suggestive of a benign breast tumor."
]

MALIGNANT_PROMPTS = [
    "A medical breast mammogram showing an irregularly shaped, spiculated mass suggestive of a malignant breast tumor.",
    "A medical breast mammogram showing an irregular, spiculated mass suggestive of a malignant breast tumor.",
    "A mammogram displaying a poorly defined, lobulated mass indicative of a malignant breast tumor.",
    "A breast imaging study revealing an irregularly shaped, invasive mass likely representing a malignant breast tumor.",
    "A mammogram showing a dense, spiculated mass suggestive of a malignant breast tumor.",
    "A breast scan identifying a poorly circumscribed, irregular mass suggestive of a malignant breast tumor.",
    "A medical breast mammogram displaying a heterogeneous, irregular mass consistent with a malignant tumor.",
    "A breast imaging scan showing a spiculated, invasive mass indicative of a malignant breast tumor.",
    "A mammogram revealing an ill-defined, irregular mass likely representing a malignant breast tumor.",
    "A breast imaging study showing an irregular, spiculated mass suggestive of a malignant breast tumor.",
    "A medical mammogram detecting a poorly defined, lobulated mass characteristic of a malignant breast tumor.",
    "A mammogram displaying an irregular, non-homogeneous mass suggestive of a malignant breast tumor.",
    "A breast scan showing a spiculated, invasive mass likely indicating a malignant breast tumor.",
    "A breast mammogram identifying an irregular, dense mass consistent with a malignant tumor.",
    "A medical mammogram revealing a spiculated, invasive mass suggestive of a malignant breast tumor.",
    "A breast imaging study showing a poorly marginated, irregular mass indicative of a malignant tumor.",
    "A mammogram displaying an irregularly shaped, spiculated mass suggestive of a malignant breast tumor.",
    "A breast scan showing an ill-defined, invasive mass likely representing a malignant breast tumor.",
    "A medical breast mammogram detecting a spiculated, irregular mass characteristic of a malignant tumor.",
    "A breast imaging study revealing a poorly circumscribed, irregular mass indicative of a malignant breast tumor.",
    "A mammogram displaying an irregular, invasive mass suggestive of a malignant breast tumor."
]

# ==================== Model Settings ====================
DEFAULT_BIOMEDCLIP_CHECKPOINT = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# ==================== Loss-specific Hyperparameters ====================
LOSS_CONFIGS = {
    'clip': {
        'temperature': 1.0,
    },
    'siglip': {
        'temperature': 1.0,
        'bias': 0.0,
    },
    'hnl': {
        'temperature': 1.0,
        'beta1': 1.0,
        'beta2': 1.0,
        'alpha': 0.0,
    },
    'mgca': {
        'emb_dim': 128,
        'hidden_dim': 2048,
        'num_prototypes': 500,
        'softmax_temperature': 0.07,
        'local_temperature': 0.1,
        'proto_temperature': 0.2,
        'lambda_1': 1.0,  # global loss weight
        'lambda_2': 1.0,  # local loss weight
        'lambda_3': 1.0,  # prototype loss weight
        'epsilon': 0.05,
        'sinkhorn_iterations': 3,
        'bidirectional': True,
        'use_local_atten': False,
        'num_heads': 1,
    },
    'gloria': {
        'temp_global': 10.0,
        'temp_local1': 4.0,  # attention temperature
        'temp_local2': 5.0,  # similarity temperature
        'temp_local3': 10.0,  # contrastive temperature
        'lambda_global': 1.0,
        'lambda_local': 1.0,
        'local_agg': 'sum',  # 'sum' or 'mean'
    },
    'dpo': {
        'alpha': 1.0,  # scaling factor for cosine similarity
        'beta': 10.0,  # DPO temperature (higher = stronger preference)
        'ref_checkpoint': '/home/jaey00ns/MedCLIP-SAMv2-main/saliency_maps/model/pytorch_model_medclipsam.bin',
    },
    'cliprefine': {
        'temperature': 1.0,
        'lambda_rand': 1.0,  # weight for random feature regularization
        'strategy': 'std_sample',  # std_sample, uniform_sample, precomputed_sample, uniform_fixed, precomputed_fixed
        'share_random_feat': True,  # whether to use same random feature for image and text
        'mu': 0.0,  # mean for random distribution
        'sigma': 1.0,  # std/upper bound for random distribution
        'precomputed_stats': None,  # path to .npz file with precomputed statistics (optional)
        'regularization_decay': False,  # whether to decay regularization weight over training
        'max_iteration': None,  # required if regularization_decay=True
    }
}

# ==================== Training Hyperparameters ====================
DEFAULT_TRAINING_CONFIG = {
    'batch_size': 8,
    'epochs': 50,
    'lr': 2e-5,
    'weight_decay': 0.05,
    'num_workers': 4,
    'warmup_epochs': 20,
    'initial_lr': 1e-8,
    'seed': 42,
    'max_length': 77,  # tokenizer max length
}
