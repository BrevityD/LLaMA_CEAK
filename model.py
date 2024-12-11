import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UpstreamNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weights=None):
        """
        Initializes the embedding layer.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            pretrained_weights (torch.Tensor or None): Pretrained embedding weights. If None, initialize randomly.
        """
        super(UpstreamNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_weights is not None:
            # Load pretrained weights into the embedding layer
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = True  # Allow finetuning if needed

    def forward(self, inputs):
        """
        Forward pass through the embedding layer.
        
        Args:
            input_ids (torch.Tensor): Input tensor of token IDs.
            
        Returns:
            torch.Tensor: Embedded representations.
        """
        input_ids, input_ratio = inputs
        input_tensor = []
        for input_id in input_ids:
            input_id = input_id.to(device)
            input_tensor.append(torch.mean(self.embedding(input_id).squeeze(), dim=0))
        # logger.debug(f"input tensor: {len(input_tensor)}, {input_tensor[0].shape}")
        input_tensor = torch.stack(input_tensor, dim=0)
        input_tensor = input_tensor.to(device)
        assert input_tensor.shape[0] == input_ratio.shape[-1], f"id: {input_tensor.shape}, ratio: {input_ratio.shape}"

        input_embedding = torch.matmul(input_ratio, input_tensor)

        return input_embedding


class DownstreamNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initializes a simple MLP for regression.
        
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
        """
        super(DownstreamNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()  # Ensures the output is positive

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output with positive values.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softplus(x)

class CEAK_Llama(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_weights=None):
        super(CEAK_Llama, self).__init__()
        self.upstream = UpstreamNetwork(vocab_size, embedding_dim, pretrained_weights)
        self.downstream = DownstreamNetwork(embedding_dim, hidden_dim)
    
    def forward(self, input_ids):
        embedded = self.upstream(input_ids)
        output = self.downstream(embedded)
        return output

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        """
        Dataset for loading data from a DataFrame.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing the data.
            input_column (str): Column name for the input sequences.
            target_column (str): Column name for the target values.
        """
        self.data = dataframe
        self.input_column = ["electrolyte 1 - smiles", "electrolyte 2 - smiles", "electrolyte 3 - smiles", "electrolyte 4 - smiles", "electrolyte 5 - smiles", "electrolyte 6 - smiles", "electrolyte 7 - smiles"]
        self.ratio_column = ["electrolyte 1 - %", "electrolyte 2 - %", "electrolyte 3 - %", "electrolyte 4 - %", "electrolyte 5 - %", "electrolyte 6 - %", "electrolyte 7 - %"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data.iloc[idx][self.input_column]
        input_ids = []
        for i in inputs:
            input_ids.append(
                self.tokenizer.encode(str(i), return_tensors="pt")
            )
        target = self.data.iloc[idx]["lce"]
        
        input_ratio = torch.tensor(list(map(float, list(self.data.iloc[idx][self.ratio_column]))), dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        assert len(input_ids) == input_ratio.shape[-1], f"id: {len(input_ids)}, ratio: {input_ratio.shape}"

        return (input_ids, input_ratio), target

def train_model(model, dataloader, criterion, optimizer, num_epochs=5, save_dir="checkpoints"):
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
            input_ids, input_ratio = inputs
            input_ratio = input_ratio.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model((input_ids, input_ratio))

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
        checkpoint_path = os.path.join(save_dir, f"llama_rand_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")
