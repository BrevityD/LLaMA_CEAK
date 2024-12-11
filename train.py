import os
import json
import time

import torch
from loguru import logger

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, dataloader, criterion, optimizer, num_epochs=5, save_dir="ckpts"):
    """
    Train the model on the given DataLoader.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the training on.
        num_epochs (int): Number of epochs to train.
    """
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_step = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device).squeeze()
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # logger.info(f"output: {outputs.shape}, {outputs}")
            # logger.info(f"target: {targets.shape}, {targets}")
            loss = criterion(outputs.squeeze(-1), targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_step += 1
            if epoch_step % 50 == 0:
                logger.debug(f"Step {epoch_step}, Loss {epoch_loss / epoch_step:.4f}")

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Step: {epoch_step}, Loss: {epoch_loss / len(dataloader):.4f}")
        checkpoint_path = os.path.join(save_dir, f"llama_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from loguru import logger
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from datasets import CEAKDataset
    from model import CEAK_Llama

    tokenizer = AutoTokenizer.from_pretrained("/home/G01-A100-20240605/pretrained_models/Meta-Llama-3-8B-Instruct")
    pretrained_model = AutoModelForCausalLM.from_pretrained("/home/G01-A100-20240605/pretrained_models/Meta-Llama-3-8B-Instruct")

    train_args = {}

    train_args["vocab_size"], train_args["embedding_dim"] = pretrained_model.get_input_embeddings().weight.shape
    train_args["hidden_dim"] = train_args["embedding_dim"] // 4
    
    train_args["batch_size"] = 1
    train_args["num_epochs"] = 10
    train_args["learning_rate"] = 0.001
    train_args["is_freezed"]=True

    logger.debug(f"vocab_size is {train_args['vocab_size']}, embedding_dim is {train_args['embedding_dim']}, hidden_dim is {train_args['hidden_dim']}")
    
    with open("ckpts/instructinit/train_args.json", "w") as f:
        json.dump(train_args, f, indent=4)
    with open("ckpts/randinit/train_args.json", "w") as f:
        json.dump(train_args, f, indent=4)

    net_instructinit_freezed = CEAK_Llama(
        vocab_size=train_args["vocab_size"],
        embedding_dim=train_args["embedding_dim"],
        hidden_dim=train_args["hidden_dim"],
        pretrained_weights=pretrained_model.get_input_embeddings().weight,
        is_freezed=train_args["is_freezed"]
    )
    net_randinit_freezed = CEAK_Llama(
        vocab_size=train_args["vocab_size"],
        embedding_dim=train_args["embedding_dim"],
        hidden_dim=train_args["hidden_dim"],
        pretrained_weights=None,
        is_freezed=train_args["is_freezed"]
    )

    del pretrained_model
    torch.cuda.empty_cache()

    criterion = nn.MSELoss()  # For regression
    optimizer = optim.Adam(net_instructinit_freezed.parameters(), lr=train_args["learning_rate"])

    dataset = CEAKDataset(
        dataframe = pd.read_csv("./data/ceak_datasets.csv"),
        tokenizer = tokenizer
        )
    dataloader = DataLoader(dataset, batch_size=train_args["batch_size"], shuffle=True)

    train_model(
        model=net_instructinit_freezed,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=train_args["num_epochs"],
        save_dir="ckpts/instructinit"
        )
    
    dataset = CEAKDataset(
        dataframe = pd.read_csv("./data/ceak_datasets.csv"),
        tokenizer = tokenizer
        )
    dataloader = DataLoader(dataset, batch_size=train_args["batch_size"], shuffle=True)
    optimizer = optim.Adam(net_instructinit_freezed.parameters(), lr=train_args["learning_rate"])
    
    train_model(
        model=net_randinit_freezed,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=train_args["num_epochs"],
        save_dir="ckpts/randinit"
        )