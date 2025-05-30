"""Model training utilities for LLaMA-CEAK.

This module provides functions for:
- Training the CEAK_Llama model
- Saving checkpoints and logs
- Tensorboard integration
- Training progress monitoring
"""

import os
import json
from typing import Optional

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

def train_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    save_dir: Optional[str] = None
) -> float:
    """Train the CEAK_Llama model on the given DataLoader.
    
    Args:
        model: CEAK_Llama model to train
        dataloader: DataLoader for training data
        criterion: Loss function (typically MSELoss)
        optimizer: Optimizer (typically Adam)
        device: Device to run training on
        num_epochs: Number of training epochs
        save_dir: Directory to save checkpoints and logs (optional)
        
    Returns:
        float: Final epoch loss
    """
    model.to(device)
    model.train()
    writer = SummaryWriter(save_dir+"/tensorboard")
    total_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_step = 0
        losses = []
        for inputs, targets in dataloader:
            if torch.is_tensor(inputs):
                inputs = inputs.to(device).squeeze()
            elif isinstance(inputs, list):
                inputs = [i[0] for i in inputs]
            targets = targets.to(device)

            # Forward pass
            # logger.debug(f"output: {type(inputs)}, content: {inputs}")

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
            total_step += 1
            writer.add_scalar("Loss/train", loss.item(), total_step)
            if epoch_step % 10 == 0:
                losses.append(str(loss.item()))
            if epoch_step % 50 == 0:
                lr = [group['lr'] for group in optimizer.param_groups]
                logger.debug(f"Step {epoch_step}, Loss {epoch_loss / epoch_step:.4f}, learning rate {lr}")

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Step: {epoch_step}, Loss: {epoch_loss / len(dataloader):.4f}")
        if save_dir:
            with open(os.path.join(save_dir, "loss.log"), "a") as wf:
                wf.write("\n".join(losses)+"\n")
            checkpoint_path = os.path.join(save_dir, f"llama_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model saved to {checkpoint_path}")

    return epoch_loss / len(dataloader)

if __name__ == "__main__":
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from loguru import logger
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from llamaceak.datasets import CEAKDataset
    from llamaceak.model import CEAK_Llama

    model_id = "/home/~/pretrained_models/Llama-3.2-1B-Instruct/"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_id)

    train_args = {}

    train_args["model_id"] = model_id
    train_args["vocab_size"], train_args["embedding_dim"] = pretrained_model.get_input_embeddings().weight.shape
    train_args["hidden_dim"] = train_args["embedding_dim"] // 4
    
    train_args["pooling"] = 8 # NEEDED None
    train_args["batch_size"] = 1
    train_args["num_epochs"] = 10
    train_args["learning_rate"] = 0.001
    train_args["is_freezed"]=False # NEEDED
    train_args["dataset_path"] = "./data/ceak_datasets_sub.csv"
    save_dir = "ckpts/llama-1B-fd-p-uf-lr13" # NEEDED llama-{1B}-{fd/od}-{p/n}-{f/uf}-{lr14}

    logger.debug(f"vocab_size is {train_args['vocab_size']}, embedding_dim is {train_args['embedding_dim']}, hidden_dim is {train_args['hidden_dim']}")
    
    with open(save_dir+"/train_args.json", "w") as f:
        json.dump(train_args, f, indent=4)

    net_instructinit_freezed = CEAK_Llama(
        vocab_size=train_args["vocab_size"],
        embedding_dim=train_args["embedding_dim"],
        hidden_dim=train_args["hidden_dim"],
        pretrained_weights=pretrained_model.get_input_embeddings().weight,
        is_freezed=train_args["is_freezed"],
        pooling=train_args["pooling"]
    )

    with open(save_dir+"/model_arc.txt", "w") as wf:
        wf.write(str(net_instructinit_freezed))
    del pretrained_model
    torch.cuda.empty_cache()

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
