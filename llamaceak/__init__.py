import torch

from .datasets import CEAKDataset
from .eval import eval_model, eval_model_onval
from .model import CEAK_Llama
from .train import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["CEAKDataset", "eval_model", "eval_model_onval", "CEAK_Llama", "train_model"]