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
    """Replace the upstream network of CEAK with the embedding layer of LLAMA.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_weights=None, is_freezed=False, pooling:int=None):
        super(CEAK_Llama, self).__init__()
        self.upstream = EmbeddingNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_weights=pretrained_weights,
            is_freezed=is_freezed
            )
        self.pooling = pooling
        if self.pooling:
            assert(isinstance(pooling, int))
            embedding_dim //= pooling
            hidden_dim //= pooling
        self.downstream = DownstreamMLP(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim
            )
    
    def forward(self, input_ids):
        embedded = self.upstream(input_ids)
        if self.pooling:
            embedded = F.avg_pool1d(embedded, kernel_size=self.pooling, stride=self.pooling)
        output = self.downstream(embedded)
        return output

class Llama_serial(nn.Module):
    """The first few layer of llama model
    """
    def __init__(self, model_path, n_layer, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        super(Llama_serial, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        whole_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device=device
        self.embedding_layer = whole_model.get_input_embeddings().to(self.device)
        self.model = nn.ModuleList(whole_model.model.layers[:n_layer]).to(self.device)
        self.past_key_values = None

        
    def forward(self, input_seq):
        from transformers import DynamicCache
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(input_seq, return_tensors="pt", max_length=128, padding="max_length").input_ids.to(self.device)
            inputs_emb = self.embedding_layer(input_ids).to(self.device)
            output = inputs_emb.clone().to(self.device)
            for layer in self.model[1:]:
                if self.past_key_values == None:
                    self.past_key_values = DynamicCache()
                else:
                    self.past_key_values = DynamicCache.from_legacy_cache(self.past_key_values)
                self.past_seen_tokens = self.past_key_values.get_seq_length() if self.past_key_values is not None else 0
                cache_position = torch.arange(self.past_seen_tokens, self.past_seen_tokens + inputs_emb.shape[1], device=inputs_emb.device)
                position_ids = cache_position.unsqueeze(0)
                output = layer(output, position_ids=position_ids)[0]
        return output[:, -1, :].reshape(1,-1)

class CEAK_Llama_Plus(nn.Module):
    """Use the entire LLAMA model (excluding the last few layers) to obtain an embedding for each input sequence (generated using a fixed prompt template). Then, utilize an MLP network to make predictions.
    """
    def __init__(self, n_layer, embedding_dim, hidden_dim, pooling:int=None, device="cuda", model_path=""):
        super(CEAK_Llama_Plus, self).__init__()
        self.upstream = Llama_serial(
            model_path=model_path,
            n_layer=n_layer,
            device=device
            )
        self.pooling = pooling
        embedding_dim *= 7
        if self.pooling:
            assert(isinstance(pooling, int))
            embedding_dim //= pooling
            hidden_dim //= pooling
        self.downstream = DownstreamMLP(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim
            ).to(device)

    def forward(self, input_seq):
        embedded = self.upstream(input_seq)
        if self.pooling:
            embedded = F.avg_pool1d(embedded, kernel_size=self.pooling, stride=self.pooling)
        output = self.downstream(embedded)
        return output
