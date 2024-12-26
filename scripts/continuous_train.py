import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

from llamaceak.datasets import CEAKDataset
from llamaceak.model import CEAK_Llama
from llamaceak.train import train_model

import pandas as pd

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "/home/~/pretrained_models/Llama-3.2-1B-Instruct/"
# model_id = "/mnt/afs/~/chkpts/sft/llamafy-llama-3.2-1b-instruct_sft_full_241104_v6/"
ckpt = "/home/~/llama_ceak/llama-ceak/ckpts/temp"
# ckpt = "/mnt/afs/~/ckpts/llama-ceak/llama-1B-od-p-uf-lr14-c"
model_path = ckpt+"/llama_epoch_1.pth"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

with open(os.path.join(ckpt, "train_args.json"), "r") as f:
    train_args = json.load(f)

train_args["model_id"] = model_id
train_args["num_epochs"] = 50
train_args["learning_rate"] = 0.0000003
train_args["dataset_path"] = "./data/ceak_datasets_sub.csv"
save_dir = "/mnt/afs/dzj/ckpts/llama-ceak/llama-1B-od-p-uf-lr15" # NEEDED llama-{1B}-{fd/od}-{p/n}-{f/uf}-{lr14}

logger.debug(f"vocab_size is {train_args['vocab_size']}, embedding_dim is {train_args['embedding_dim']}, hidden_dim is {train_args['hidden_dim']}")

with open(save_dir+"/train_args.json", "w") as f:
    json.dump(train_args, f, indent=4)

model = CEAK_Llama(
    vocab_size=train_args["vocab_size"],
    embedding_dim=train_args["embedding_dim"],
    hidden_dim=train_args["hidden_dim"],
    pooling=train_args["pooling"]
)
state_dict = torch.load(model_path, weights_only=True)
model.load_state_dict(state_dict)

with open(save_dir+"/model_arc.txt", "w") as wf:
    wf.write(str(model))

criterion = nn.MSELoss()  # For regression
optimizer = optim.Adam(model.parameters(), lr=train_args["learning_rate"])

dataset = CEAKDataset(
    dataframe = pd.read_csv(train_args["dataset_path"]),
    tokenizer = tokenizer
    )
dataloader = DataLoader(dataset, batch_size=train_args["batch_size"], shuffle=True)

train_model(
    model=model,
    dataloader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=train_args["num_epochs"],
    save_dir=save_dir
    )