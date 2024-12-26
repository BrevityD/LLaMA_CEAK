"""This script loads a single layer of a 70B model, as the full model is too large to fit on the GPU."""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from safetensors.torch import load_file
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llamaceak.datasets import CEAKDataset
from llamaceak.model import CEAK_Llama
from llamaceak.train import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "/mnt/afs/~/pretrained_models/Llama-3.3-70B-Instruct"
layer_file = "/model-00001-of-00030.safetensors"

tokenizer = AutoTokenizer.from_pretrained(model_id)
state_dict = load_file(model_id+layer_file)
pretrained_weights = state_dict["model.embed_tokens.weight"]  # tensor of [128256, 8192]

train_args = {}

train_args["model_id"] = model_id
train_args["vocab_size"], train_args["embedding_dim"] = pretrained_weights.shape
train_args["hidden_dim"] = train_args["embedding_dim"] // 4

train_args["pooling"] = 8 # NEEDED None
train_args["batch_size"] = 1
train_args["num_epochs"] = 20
train_args["learning_rate"] = 0.001
train_args["is_freezed"]=True # NEEDED
train_args["dataset_path"] = "./data/ceak_datasets.csv"
save_dir = "ckpts/llama-70B-od-p-f-lr13" # NEEDED llama-{1B}-{fd/od}-{p/n}-{f/uf}-{lr14}

logger.debug(f"vocab_size is {train_args['vocab_size']}, embedding_dim is {train_args['embedding_dim']}, hidden_dim is {train_args['hidden_dim']}")

with open(save_dir+"/train_args.json", "w") as f:
    json.dump(train_args, f, indent=4)

net_instructinit_freezed = CEAK_Llama(
    vocab_size=train_args["vocab_size"],
    embedding_dim=train_args["embedding_dim"],
    hidden_dim=train_args["hidden_dim"],
    pretrained_weights=pretrained_weights,
    is_freezed=train_args["is_freezed"],
    pooling=train_args["pooling"]
)

with open(save_dir+"/model_arc.txt", "w") as wf:
    wf.write(str(net_instructinit_freezed))
del pretrained_weights

criterion = nn.MSELoss()  # For regression
optimizer = optim.Adam(net_instructinit_freezed.parameters(), lr=train_args["learning_rate"])

dataset = CEAKDataset(
    dataframe = pd.read_csv(train_args["dataset_path"]),
    tokenizer = tokenizer
    )
dataloader = DataLoader(dataset, batch_size=train_args["batch_size"], shuffle=True)

train_model(
    model=net_instructinit_freezed,
    dataloader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=train_args["num_epochs"],
    save_dir=save_dir,
    device=device
    )