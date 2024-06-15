import math
import torch
import torch.nn as nn
import torchtune.modules as torchmodule
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self,d_model,dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)

class TransformerBlock(nn.Module):

    def __init__(self,d_model,context_length,num_heads,dropout):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads  # head size should be divisible by d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.embedding=torchmodule.RotaryPositionalEmbeddings(dim=self.head_size,max_seq_len=self.context_length)
        self.multi_head_attention_layer = nn.MultiheadAttention(d_model, num_heads,batch_first=True)
        self.att_mask=torch.tril(torch.ones((self.context_length,self.context_length)))
        self.att_mask=self.att_mask.masked_fill(self.att_mask==0,float("-inf"))
        self.att_mask=self.att_mask.masked_fill(self.att_mask==1,0)
        self.feed_forward_layer = FeedForward(self.d_model,self.dropout)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        B,S,D=x.shape
        q=k=x.view((B,S,self.num_heads,self.head_size))
        q=self.embedding(q)
        k=self.embedding(k)
        q=q.view((B,S,D))
        k=k.view((B,S,D))
        x , _ = self.multi_head_attention_layer(q,k,x,attn_mask=self.att_mask)
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # Residual connection
        return x
    
class TransformerLanguageModel(nn.Module):
    def __init__(self,d_model,context_length,num_heads,num_blocks,dropout,max_token_value):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        # Set up token embedding look-up table
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1, embedding_dim=self.d_model)

        # Run all the transformer blocks
        # Different from original paper, here we add a final layer norm after all the blocks
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(self.d_model,self.context_length,self.num_heads,self.dropout) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_lookup_table(idx)
        x = self.transformer_blocks(x)
        # The "logits" are the output values of our model before applying softmax
        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            idx_crop = idx[:, -self.context_length:]
            # Get predictions
            logits, loss = self(idx_crop)
            # Get the last time step from logits where the dimensions of the logits are (B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
