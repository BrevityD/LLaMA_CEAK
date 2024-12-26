import torch
from torch.utils.data import Dataset

class CEAKDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        """
        Dataset for loading data from a DataFrame.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing the data.
            input_column (str): Column name for the input sequences.
            target_column (str): Column name for the target values.
        """
        self.data = dataframe
        self.input_column = ["electrolyte 1 - smiles", "electrolyte 2 - smiles", "electrolyte 3 - smiles", "electrolyte 4 - smiles", "electrolyte 5 - smiles", "electrolyte 6 - smiles", "electrolyte 7 - smiles"]
        self.ratio_column = ["electrolyte 1 - %", "electrolyte 2 - %", "electrolyte 3 - %", "electrolyte 4 - %", "electrolyte 5 - %", "electrolyte 6 - %", "electrolyte 7 - %"]
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data.iloc[idx][self.input_column]
        input_ids = []
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

        target = self.data.iloc[idx]["lce"]
        target = torch.tensor(target, dtype=torch.float32)

        return input_ids, target