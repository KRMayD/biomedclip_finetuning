"""
Contrastive Loss Functions for BiomedCLIP Fine-tuning
=====================================================
"""

from .clip_loss import CLIPLoss
from .siglip_loss import SigLIPLoss
from .hnl_loss import HardNegativeLoss
from .mgca_loss import MGCALoss
from .gloria_loss import GLoRIALoss
from .cliprefine_loss import CLIPRefineLoss

try:
    from .dpo_loss import DPOLoss
except ModuleNotFoundError:
    DPOLoss = None

__all__ = [
    'CLIPLoss',
    'SigLIPLoss', 
    'HardNegativeLoss',
    'MGCALoss',
    'GLoRIALoss',
    'DPOLoss',
    'CLIPRefineLoss',
]
