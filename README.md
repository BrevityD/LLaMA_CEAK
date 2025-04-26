# LLaMA-CEAK: Integrating LLaMA Embeddings with CEAK Model

## Project Description
This project replaces the upstream network of CEAK (Chemical Electrolyte Analysis Kit) with components from LLaMA language model. It provides tools for:
- Training custom models with LLaMA embeddings
- Evaluating model performance
- Predicting electrolyte properties

## Features
- Integration of LLaMA embeddings with CEAK architecture
- Support for both ID-based and prompt-based input formats
- Flexible configuration options:
  - Pretrained weight loading
  - Embedding layer freezing
  - Pooling operations
  - Partial LLaMA model integration

## Installation

### Prerequisites
- Transformers <=4.46.1

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/BrevityD/LLaMA_CEAK.git
cd LLaMA_CEAK
```

2. Install dependencies using Poetry:
```bash
poetry install
```

### Configuration
Modify `pyproject.toml` for project-specific settings and dependencies.

## Usage

### Training Options
The project provides several training scripts:

1. **Standard Training**: `scripts/train_fewlayer.py`
2. **Continuous Training**: `scripts/continuous_train.py`
3. **K-Fold Validation**: `scripts/kfold_train.py`
4. **Large Model Training**: `scripts/train_70B.py`

### Basic Training Example

```python
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
```

### Evaluation
```python
from transformers import AutoTokenizer

from llamaceak.eval import eval_model
import pandas as pd  # Required to be imported last for compatibility

testset_file="./data/ceak.csv"
model_folder="./ckpts/dpo-v1v1-1B-fd-p-f-lr13-5fd"
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
```

## Model Checkpoints
Checkpoints follow a standardized naming convention:
```
llama-{MODEL_SIZE}-{DATASET}-{POOLING}-{FREEZE}-{LR}
```
Where:
- `MODEL_SIZE`: Model parameter count (e.g., 1B, 70B)
- `DATASET`: `fd` (filtered) or `od` (original)
- `POOLING`: `p` (pooled) or `n` (no pooling)
- `FREEZE`: `f` (frozen) or `uf` (unfrozen)
- `LR`: Learning rate (e.g., lr13 for 1e-3)

## Project Structure
```
.
├── llamaceak/            # Core package
│   ├── __init__.py       # Module exports
│   ├── model.py          # Model architectures
│   ├── datasets.py       # Data loading
│   ├── train.py          # Training utilities
│   └── eval.py           # Evaluation utilities
├── scripts/              # Training scripts
│   ├── continuous_train.py
│   ├── eval_llama.py
│   ├── kfold_train.py
│   ├── train_70B.py
│   └── train_fewlayer.py
├── utils/                # Utility scripts
│   ├── subdata.py        # Data filtering
│   └── test_dataloader.py
├── pyproject.toml        # Project configuration
└── README.md             # Project documentation
```

## API Documentation
For detailed API documentation, refer to the docstrings in each module file.
