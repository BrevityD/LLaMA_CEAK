import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llamaceak.datasets import CEAKDataset

model_id = ""

tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = CEAKDataset(
    dataframe = pd.read_csv(".csv"),
    tokenizer = tokenizer
    )
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for i in dataloader:
    print(i[0].shape)
    break