"""
Model wrappers for BiomedCLIP
==============================
"""

from .biomedclip_wrapper import (
    load_biomedclip,
    load_model,
    load_openclip,
    get_biomedclip_features,
    get_biomedclip_features_mgca,
)

__all__ = [
    'load_biomedclip',
    'load_model',
    'load_openclip',
    'get_biomedclip_features',
    'get_biomedclip_features_mgca',
]
