import os
import json

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, dataloader, criterion, optimizer, num_epochs=5, save_dir=None, device=device):
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
    # writer = SummaryWriter(save_dir+"/tensorboard")
    total_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_step = 0
        losses = []
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
            total_step += 1
            # writer.add_scalar("Loss/train", loss.item(), total_step)
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

    from datasets import CEAKDataset
    from model import CEAK_Llama

    # model_id = "/home/G01-A100-20240605/pretrained_models/Meta-Llama-3-8B-Instruct/"
    model_id = "/home/G01-A100-20240605/pretrained_models/Llama-3.2-1B-Instruct/"
    # model_id = "/mnt/afs/dwc/chkpts/sft/llamafy-llama-3.2-1b-instruct_sft_full_241104_v6/"

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
        save_dir=save_dir
        )