import json
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from loguru import logger
import pandas as pd
from torch.utils.data import DataLoader

from llamaceak.datasets import CEAKDataset
from llamaceak.model import CEAK_Llama_Plus
from llamaceak.train import train_model

device=torch.device("cuda:7")

warnings.simplefilter(action='ignore', category=FutureWarning)

train_args = {}

train_args["model_id"] = "/home/G01-A100-20240605/pretrained_models/Llama-3.2-1B-Instruct/"# "/mnt/afs/dwc/chkpts/dpo/sft_full_241104_v1-dpo_full_241220_v4"
train_args["vocab_size"], train_args["embedding_dim"] = 128256, 2048
train_args["hidden_dim"] = train_args["embedding_dim"] // 4

train_args["pooling"] = 8 # NEEDED None
train_args["batch_size"] = 1
train_args["num_epochs"] = 10
train_args["learning_rate"] = 0.001
train_args["is_freezed"]=None # NEEDED
train_args["dataset_path"] = "./data/ceak_datasets.csv"
train_args["n_layer"] = 4
save_dir = "ckpts/llama-1B-od-p-f-ly4-lr13" # NEEDED llama-{1B}-{fd/od}-{p/n}-{f/uf}-{ly4}-{lr14}

logger.debug(f"vocab_size is {train_args['vocab_size']}, embedding_dim is {train_args['embedding_dim']}, hidden_dim is {train_args['hidden_dim']}")

with open(save_dir+"/train_args.json", "w") as f:
    json.dump(train_args, f, indent=4)

model = CEAK_Llama_Plus(n_layer=train_args["n_layer"], embedding_dim=train_args["embedding_dim"], hidden_dim=train_args["hidden_dim"], pooling=train_args["pooling"], device=device, model_path=train_args["model_id"])

with open(save_dir+"/model_arc.txt", "w") as wf:
    wf.write(str(model))

criterion = nn.MSELoss()  # For regression
optimizer = optim.Adam(model.parameters(), lr=train_args["learning_rate"])

dataset = CEAKDataset(
    dataframe = pd.read_csv(train_args["dataset_path"]),
    tokenizer = None,
    is_id=False
)
dataloader = DataLoader(dataset, batch_size=train_args["batch_size"], shuffle=True)

train_model(
    model=model,
    dataloader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=train_args["num_epochs"],
    save_dir=save_dir,
    device=device
    )