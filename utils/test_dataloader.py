"""Test script for verifying CEAKDataset and DataLoader functionality.

This script:
- Loads a pretrained tokenizer
- Creates a CEAKDataset instance
- Tests the DataLoader output shape
"""
import pandas as pd  # type: ignore
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llamaceak.datasets import CEAKDataset

def main() -> None:
    """Main function to test dataloader functionality."""
    model_id = "/home/~/pretrained_models/Meta-Llama-3-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    dataset = CEAKDataset(
        dataframe=pd.read_csv("./data/ceak_datasets.csv"),
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for batch in dataloader:
        print(batch[0].shape)  # Print shape of first batch
        break

if __name__ == "__main__":
    main()
