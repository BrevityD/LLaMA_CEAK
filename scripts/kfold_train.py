import os

import json
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from llamaceak.eval import eval_model_onval
from llamaceak.model import CEAK_Llama
from llamaceak.train import train_model

def k_fold_cross_validation(
        dataset,
        train_args,
        pretrained_model,
        model_class=CEAK_Llama, 
        k=5, 
        batch_size=1,
        epochs=10,
        learning_rate=0.001,
        device=torch.device('cuda'), 
        save_dir='ckpts',
        ):
    kf = KFold(n_splits=k, shuffle=True)
    best_val_accuracy = 0.0
    best_checkpoint = None
    
    os.makedirs(save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # Split the dataset into training and validation sets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        model = model_class(
            vocab_size=train_args["vocab_size"],
            embedding_dim=train_args["embedding_dim"],
            hidden_dim=train_args["hidden_dim"],
            pretrained_weights=pretrained_model.get_input_embeddings().weight,
            is_freezed=train_args["is_freezed"],
            pooling=train_args["pooling"]
        ).to(device)
        criterion = nn.MSELoss()  # For regression
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model
        best_loss = float('inf')
        best_fold_checkpoint = None
        for epoch in range(epochs):  # Use a suitable number of epochs            
            # Training phase
            train_loss = train_model(
                model=model, 
                dataloader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                num_epochs=1,
                device=device
                )
            # Validation phase
            val_loss = eval_model_onval(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device
                )
            logger.info(f"Fold {fold+1}/{k}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss}")
            
            # Save the best model based on validation accuracy
            if val_loss < best_loss:
                best_loss = val_loss
                best_fold_checkpoint = model  # Save the best model checkpoint

        # Save the best model of this fold
        checkpoint_path = os.path.join(save_dir, f"best_model_fold{fold+1}.pth")
        torch.save(best_fold_checkpoint.state_dict(), checkpoint_path)
        logger.info(f"Best checkpoint for fold {fold+1} saved to {checkpoint_path}")
    
    return best_checkpoint

if __name__ == "__main__":
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from llamaceak.datasets import CEAKDataset

    model_id = "/mnt/afs/~/chkpts/dpo/sft_full_241104_v1-dpo_full_241210_v1/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_id)

    train_args = {}

    train_args["model_id"] = model_id
    train_args["vocab_size"], train_args["embedding_dim"] = pretrained_model.get_input_embeddings().weight.shape
    train_args["hidden_dim"] = train_args["embedding_dim"] // 4
    
    train_args["pooling"] = 8 # NEEDED None
    train_args["batch_size"] = 1
    train_args["num_epochs"] = 10
    train_args["learning_rate"] = 0.0005
    train_args["is_freezed"]=True # NEEDED
    train_args["dataset_path"] = "./data/ceak_datasets_sub.csv"
    train_args["k_folds"] = 5
    save_dir = "ckpts/temp" # NEEDED llama-{1B}-{fd/od}-{p/n}-{f/uf}-{lr14}-{5fd}

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
    # del pretrained_model
    # torch.cuda.empty_cache()

    criterion = nn.MSELoss()  # For regression

    dataset = CEAKDataset(
        dataframe = pd.read_csv(train_args["dataset_path"]),
        tokenizer = tokenizer
        )
    k_fold_cross_validation(
        dataset=dataset,
        train_args=train_args,
        pretrained_model=pretrained_model,
        model_class=CEAK_Llama,
        k=train_args["k_folds"],
        batch_size=train_args["batch_size"],
        epochs=train_args["num_epochs"],
        learning_rate=train_args["learning_rate"],
        device=torch.device('cuda'), 
        save_dir=save_dir
        )