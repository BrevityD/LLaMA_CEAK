import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import CEAKDataset

model_id = "/home/G01-A100-20240605/pretrained_models/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = CEAKDataset(
    dataframe = pd.read_csv("./data/ceak_datasets.csv"),
    tokenizer = tokenizer
    )
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for i in dataloader:
    print(i[0].shape)
    break