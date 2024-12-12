import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class EmbeddingNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weights=None, is_freezed=False):
        """
        Initializes the embedding layer.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            pretrained_weights (torch.Tensor or None): Pretrained embedding weights. If None, initialize randomly.
            is_freezed (boolean): will the weight of embedding layer be freezed during training
        """
        super(EmbeddingNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_weights is not None:
            # Load pretrained weights into the embedding layer
            self.embedding.weight.data.copy_(pretrained_weights)
        if is_freezed:
            self.embedding.weight.requires_grad = False
        else:
            self.embedding.weight.requires_grad = True
        logger.debug(f"pretrain weights loaded: {pretrained_weights!=None}, layer freezed: {is_freezed}")

    def forward(self, inputs, device=torch.device('cuda')):
        """
        Forward pass through the embedding layer.
        
        Args:
            inputs (torch.Tensor): Input tensor of token IDs and their ratio
            with shape [7, max_seq_len+1], where the first column is ratio
            
        Returns:
            torch.Tensor: Embedded representations.
        """
        # Ideally I would add some assertions to check the shape of input
        # But it's more appropriate to perform such checks during data preparation
        input_tensor = [] 
        for row in inputs:
            input_ids = row[1:].int() # Exclude ratio
            # logger.debug(f"input ids: {input_ids.shape}")
            if len(input_ids) > 0:
                input_tensor.append(torch.mean(self.embedding(input_ids).squeeze(), dim=0))
        # logger.debug(f"input tensor: {len(input_tensor)}, {input_tensor[0].shape}")
        input_tensor = torch.stack(input_tensor, dim=0).to(device)
        input_ratio = inputs[:, 0].unsqueeze(0)
        assert input_tensor.shape[0] == input_ratio.shape[-1], f"id: {input_tensor.shape}, ratio: {input_ratio.shape}"
        
        return torch.matmul(input_ratio, input_tensor)

class DownstreamMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initializes a simple MLP for regression.
        
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
        """
        super(DownstreamMLP, self).__init__()
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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_weights=None, is_freezed=False):
        super(CEAK_Llama, self).__init__()
        self.upstream = EmbeddingNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_weights=pretrained_weights,
            is_freezed=is_freezed
            )
        self.downstream = DownstreamMLP(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim
            )
    
    def forward(self, input_ids):
        embedded = self.upstream(input_ids)
        output = self.downstream(embedded)
        return output
