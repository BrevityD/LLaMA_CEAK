"""Dataset classes for LLaMA-CEAK model training and evaluation.

This module contains custom PyTorch Dataset implementations for handling:

- Electrolyte composition data
- Tokenization of chemical representations
- Ratio-based input formatting
- Target value (lce) loading
"""

import torch
from torch.utils.data import Dataset

class CEAKDataset(Dataset):
    """PyTorch Dataset for electrolyte composition data with LLaMA tokenization.
    
    Handles both ID-based and prompt-based input formats for LLaMA integration.
    
    Args:
        dataframe (pd.DataFrame): Input data containing electrolyte SMILES and ratios
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text processing
        is_id (bool): Whether to use ID-based (True) or prompt-based (False) formatting
    
    Attributes:
        data (pd.DataFrame): Source data
        input_column (list[str]): Column names for electrolyte SMILES
        ratio_column (list[str]): Column names for electrolyte percentages
        tokenizer: Text tokenizer instance
        is_id: Input format flag
    """
    def __init__(self, dataframe, tokenizer, is_id=True):
        self.data = dataframe
        self.input_column = ["electrolyte 1 - smiles", "electrolyte 2 - smiles", "electrolyte 3 - smiles", "electrolyte 4 - smiles", "electrolyte 5 - smiles", "electrolyte 6 - smiles", "electrolyte 7 - smiles"]
        self.ratio_column = ["electrolyte 1 - %", "electrolyte 2 - %", "electrolyte 3 - %", "electrolyte 4 - %", "electrolyte 5 - %", "electrolyte 6 - %", "electrolyte 7 - %"]
        self.tokenizer = tokenizer

        if self.tokenizer:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.is_id = is_id


    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets a single sample from the dataset.
        
        Args:
            idx (int): Index of sample to retrieve
            
        Returns:
            tuple: (input_tensor, target_tensor) where:
                input_tensor: Either token IDs with ratios (is_id=True) or 
                             prompt strings (is_id=False)
                target_tensor: lce value as float tensor
        """
        inputs = self.data.iloc[idx][self.input_column]
        input_ids = []
        
        if self.is_id:
            for i in inputs:
                input_ids.append(
                    self.tokenizer.encode(
                        str(i),
                        max_length=128,
                        padding='max_length',
                        return_tensors="pt"
                        )
                )
            input_ids = torch.cat(input_ids, dim=0)
            input_ratio = torch.tensor(list(map(float, list(self.data.iloc[idx][self.ratio_column]))), dtype=torch.float32).unsqueeze(1)

            input_ids = torch.cat((input_ratio, input_ids), dim=-1)
            # logger.info(f"shape of datasets: {input_ids.shape}")
        else:
            prompt_template = "Express the electrolyte component {electrolyte} with a ratio of {ratio} percents using a single word:"
            for i, inp in enumerate(inputs):
                input_ids.append(
                    prompt_template.format(
                        electrolyte = str(inp),
                        ratio = float(self.data.iloc[idx][self.ratio_column][i])
                    )
                )

        target = self.data.iloc[idx]["lce"]
        target = torch.tensor(target, dtype=torch.float32)

        return input_ids, target
