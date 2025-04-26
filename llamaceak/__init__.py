"""LLaMA-CEAK: Integrating LLaMA Embeddings with CEAK Model

This module provides integration between LLaMA language model embeddings 
and the CEAK model architecture. It includes components for:

- Model definition (CEAK_Llama)
- Dataset handling (CEAKDataset) 
- Training (train_model)
- Evaluation (eval_model, eval_model_onval)

The module is designed to experiment with replacing CEAK's upstream network
with LLaMA's embedding layer while maintaining the original functionality.
"""

import torch

from .datasets import CEAKDataset
from .eval import eval_model, eval_model_onval
from .model import CEAK_Llama
from .train import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["CEAKDataset", "eval_model", "eval_model_onval", "CEAK_Llama", "train_model"]
