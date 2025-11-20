from datasets import load_dataset
import torch
from torch import nn
import math
from transformers import AutoTokenizer

def tokenize_data():
    
    tiny_stories_train = load_dataset("roneneldan/TinyStories", split="train")
    tiny_stories_val = load_dataset("roneneldan/TinyStories", split="validation")

    max_len = 128
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

    train_map = tiny_stories_train.map(tokenize_fn, batched=True)
    val_map = tiny_stories_val.map(tokenize_fn, batched=True)

    return tokenizer, train_map, val_map

tokenize_data()



def sinusoidal_position_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(seq_len, device=device).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (T, D)

def causal_tgt_mask(T, device):
    """Causal mask for nn.TransformerDecoder: shape (T, T), bool."""
    
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
    return mask

class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, d_ff=256,
                 max_len=512, attn_dropout=0.0, resid_dropout=0.0, disable_positional=False, 
                 dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.disable_positional = disable_positional
        self.max_len = max_len
        self.d_model = d_model

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        pass
