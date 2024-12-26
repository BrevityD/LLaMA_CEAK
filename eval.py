import os

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoTokenizer

from model import CEAK_Llama
import pandas as pd # 只有把pandas的import放在最后才能运行

checkpoint_folder = "./ckpts/"

def eval_model_onval(model, val_loader, criterion, device=torch.device('cuda')):
    model.eval()
    eval_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device).squeeze()
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(-1), targets)
            eval_loss += loss.item()
            total_samples += 1
        rmse = torch.sqrt(torch.tensor(eval_loss / total_samples))
    logger.info(f"the valid rmse is {rmse}")
    return rmse.tolist()
    

def eval_model(model_path, train_args, dataframe, tokenizer, device=torch.device('cuda')):
    input_columns = ["electrolyte 1 - smiles", "electrolyte 2 - smiles", "electrolyte 3 - smiles", "electrolyte 4 - smiles", "electrolyte 5 - smiles", "electrolyte 6 - smiles", "electrolyte 7 - smiles"]
    ratio_columns = ["electrolyte 1 - %", "electrolyte 2 - %", "electrolyte 3 - %", "electrolyte 4 - %", "electrolyte 5 - %", "electrolyte 6 - %", "electrolyte 7 - %"]

    model = CEAK_Llama(
        vocab_size=train_args["vocab_size"],
        embedding_dim=train_args["embedding_dim"],
        hidden_dim=train_args["hidden_dim"],
        pooling=train_args["pooling"]
    )
    state_dict = torch.load(model_path, weights_only=True)

    # Load the weights into the model
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    logger.info(f"Model {model_path.replace('.pth','').split('/')[-1]} loaded successfully!")

    mse_loss = nn.MSELoss(reduction='sum')
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for index, row in dataframe.iterrows():
            inputs = row[input_columns]
            input_ids = []
            for i in inputs:
                input_ids.append(
                    tokenizer.encode(str(i),
                                     max_length=128,
                                     padding='max_length',
                                     return_tensors="pt")
                    )
            target = row["lce"]
            target = torch.tensor([target], dtype=torch.float32)
            input_ratio = torch.tensor(list(map(float, list(row[ratio_columns]))), dtype=torch.float32).unsqueeze(1)
            input_ids = torch.cat(input_ids, dim=0)

            input_ids = torch.cat((input_ratio, input_ids), dim=-1).to(device)
            target = target.to(device)

            outputs = model(input_ids)

            # logger.debug(f"predicted value: {outputs.squeeze(-1)}, target value: {target}")

            mse = mse_loss(outputs.squeeze(-1), target)
            total_mse += mse.item()
            total_samples += 1

    rmse = torch.sqrt(torch.tensor(total_mse / total_samples))
    logger.info(f"the rmse of ckpt in {model_path} is {rmse}")
    return rmse.tolist()

if __name__ == "__main__":
    testset_file="./data/ceak_experiments_hzx_sub.csv"
    model_folder="./ckpts/dpo-v1v1-1B-fd-p-f-lr13-5fd"# "/mnt/afs/dzj/ckpts/llama-ceak/llama-1B-od-p-uf-lr15"# 
    dataframe = pd.read_csv(testset_file)
    pth_files = []
    for root, _, files in os.walk(model_folder):
        for file in files:
            if file.endswith(".pth"):
                pth_files.append(os.path.join(root, file))

    import json
    with open(os.path.join(model_folder, "train_args.json"), "r") as f:
        train_args = json.load(f)

    model_id = train_args["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    rmse = {}

    for model_file in pth_files:
        rmse.update(
            {
                model_file: eval_model(
                    model_path=model_file,
                    train_args=train_args,
                    dataframe=dataframe,
                    tokenizer=tokenizer,
                )
            }
        )
    with open(os.path.join(model_folder, "eval_result.json"), "w") as f:
        json.dump(rmse, f)
