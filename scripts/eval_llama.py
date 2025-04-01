import os
import warnings

import torch
import torch.nn as nn
from loguru import logger

from llamaceak.model import CEAK_Llama_Plus
import pandas as pd  # 只有把pandas的import放在最后才能运行

warnings.simplefilter(action='ignore', category=FutureWarning)

def eval_model(model_path, train_args, dataframe, device=torch.device('cuda')):
    input_columns = ["electrolyte 1 - smiles", "electrolyte 2 - smiles", "electrolyte 3 - smiles", "electrolyte 4 - smiles", "electrolyte 5 - smiles", "electrolyte 6 - smiles", "electrolyte 7 - smiles"]
    ratio_columns = ["electrolyte 1 - %", "electrolyte 2 - %", "electrolyte 3 - %", "electrolyte 4 - %", "electrolyte 5 - %", "electrolyte 6 - %", "electrolyte 7 - %"]

    model = CEAK_Llama_Plus(
        n_layer=train_args["n_layer"], embedding_dim=train_args["embedding_dim"], hidden_dim=train_args["hidden_dim"], pooling=train_args["pooling"], device=device
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
            prompt_template = "Express the electrolyte component {electrolyte} with a ratio of {ratio} percents using a single word:"
            for i, inp in enumerate(inputs):
                input_ids.append(
                    prompt_template.format(
                        electrolyte = str(inp),
                        ratio = float(row[ratio_columns][i])
                    )
                )
            target = row["lce"]
            target = torch.tensor([target], dtype=torch.float32)

            target = target.to(device)

            outputs = model(input_ids)

            # logger.debug(f"predicted value: {outputs.squeeze(-1)}, target value: {target}")

            mse = mse_loss(outputs.squeeze(-1), target)
            total_mse += mse.item()
            total_samples += 1

    rmse = torch.sqrt(torch.tensor(total_mse / total_samples))
    logger.info(f"the rmse of ckpt in {model_path} is {rmse}")
    return rmse.tolist()


testset_file=".csv"
model_folder=""
dataframe = pd.read_csv(testset_file)
pth_files = []
for root, _, files in os.walk(model_folder):
    for file in files:
        if file.endswith(".pth"):
            pth_files.append(os.path.join(root, file))

import json
with open(os.path.join(model_folder, "train_args.json"), "r") as f:
    train_args = json.load(f)

rmse = {}

for model_file in pth_files:
    rmse.update(
        {
            model_file: eval_model(
                model_path=model_file,
                train_args=train_args,
                dataframe=dataframe,
                device=torch.device("cuda:7")
            )
        }
    )
with open(os.path.join(model_folder, "eval_result.json"), "w") as f:
    json.dump(rmse, f)


