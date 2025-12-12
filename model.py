import math
import time
from dataclasses import dataclass, asdict

import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


# Data Loading and Tokenization
def tokenize_data(max_len=128, n_train=20000, n_val=5000):
    """
    Load and tokenize the TinyStories dataset.
    Function returns the tokenizer and tokenized train/validation datasets.
    """
    tiny_stories_train = load_dataset("roneneldan/TinyStories", split="train")
    tiny_stories_val = load_dataset("roneneldan/TinyStories", split="validation")

    tiny_stories_train = tiny_stories_train.select(range(min(n_train, len(tiny_stories_train))))
    tiny_stories_val = tiny_stories_val.select(range(min(n_val, len(tiny_stories_val))))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

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

def get_batch(ds, batch_size, device):
    """
    Yields batches of input_ids on the given device.
    """
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        yield torch.tensor(batch["input_ids"], device=device)

# Positional Encodings
def sinusoidal_position_encoding(seq_len, d_model, device):
    """
    Generating sinusoidal positional encodings.
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(seq_len, device=device).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (T, D)

def causal_tgt_mask(T, device):
    """
    Causal mask of shape (T, T), bool.
    mask[i, j] = True means pos j is masked when predicting i (future token).
    """
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
    return mask

def rotate_half(x):
    """
    x: (..., dim)
    Splits last dimension into two halves and rotates, used for rotary 
    positional embeddings.
    """
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Implements rotary positional embeddings for a single head dimension.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # (dim/2,)

    def get_cos_sin(self, seq_len, device):
        """
        Returns cos and sin tensors of shape (1, 1, seq_len, dim)
        suitable for broadcasting onto (B, n_heads, seq_len, dim).
        """
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=device)  # (T,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        cos = emb.cos()[None, None, :, :]  # (1, 1, T, dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, T, dim)
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k: (B, n_heads, T, head_dim)
    cos, sin: (1, 1, T, head_dim)
    """
    q2 = (q * cos) + (rotate_half(q) * sin)
    k2 = (k * cos) + (rotate_half(k) * sin)
    return q2, k2

# Attention block with optional rotary positional embeddings
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_rotary=False, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim, max_position_embeddings=max_len)
        else:
            self.rotary = None
        self.save_attn = False
        self.last_attn_weights = None

    def forward(self, x):
        """
        x: (B, T, D)
        Returns: (B, T, D)
        """
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary:
            cos, sin = self.rotary.get_cos_sin(T, x.device)  # (1, 1, T, head_dim)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention scores: (B, n_heads, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        mask = causal_tgt_mask(T, x.device)  # (T, T)
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float("-inf"))

        # Softmax + dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if self.save_attn:
            self.last_attn_weights = attn_weights.detach().cpu()

        # Attention output: (B, n_heads, T, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads: (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 use_rotary=False, max_len=512):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            max_len=max_len,
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, T, D)
        # Pre-norm + attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = residual + x

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + x
        return x
    
class MiniTransformerLM(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            d_model=128, 
            n_layers=2, 
            n_heads=4, 
            d_ff=256,
            max_len=512, 
            dropout=0.1,
            pos_encoding_type="sinusoidal"): # "sinusoidal"/"learned"/"rotary"/"none"
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type
        
        if pos_encoding_type == "learned":
            self.pos_embed = nn.Embedding(max_len, d_model)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_rotary=(pos_encoding_type == "rotary"),
                max_len=max_len,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed(input_ids)

        if self.pos_encoding_type == "sinusoidal":
            pe = sinusoidal_position_encoding(T, self.d_model, device)
            x = x + pe.unsqueeze(0)
        elif self.pos_encoding_type == "learned":
            positions = torch.arange(T, device=device).unsqueeze(0)
            pos_emb = self.pos_embed(positions)
            x = x + pos_emb

        mask = causal_tgt_mask(T, input_ids.device)
        memory = torch.zeros(B, 1, self.d_model, device=input_ids.device)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
    
# Training config and loop
@dataclass
class TrainConfig:
    max_len: int = 128
    batch_size: int = 16
    epochs: int = 1
    lr: float = 3e-4
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    pos_encoding_type: str = "sinusoidal"  # 'sinusoidal'/'learned'/'rotary'/'none'
    n_train: int = 20000
    n_val: int = 5000
    label: str = "baseline"

def train_one_config(cfg: TrainConfig, device=None):
    """
    Train model under the given config.
    Returns (model, logs_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Training config: {cfg.label} ===")
    print(asdict(cfg))

    tokenizer, train_ds, val_ds = tokenize_data(
        max_len=cfg.max_len, n_train=cfg.n_train, n_val=cfg.n_val
    )

    model = MiniTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
        pos_encoding_type=cfg.pos_encoding_type,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train_loss_history = []
    val_loss_history = []
    step_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        for input_ids in get_batch(train_ds, cfg.batch_size, device):
            step_start = time.perf_counter()

            x = input_ids[:, :-1]
            y = input_ids[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
            train_loss_history.append(loss.item())
            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"step {global_step} | epoch {epoch} | "
                    f"loss {loss.item():.4f}"
                )

        # Validation each epoch
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for input_ids in get_batch(val_ds, cfg.batch_size, device):
                x = input_ids[:, :-1]
                y = input_ids[:, 1:]
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        val_loss_history.append(val_loss)
        print(f"[{cfg.label}] epoch {epoch} validation loss: {val_loss:.4f}")

    total_time = time.perf_counter() - start_time

    if device.type == "cuda":
        max_mem_bytes = torch.cuda.max_memory_allocated(device)
        max_mem_mb = max_mem_bytes / (1024 ** 2)
    else:
        max_mem_mb = None

    logs = {
        "config": asdict(cfg),
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "step_times": step_times,
        "total_time_sec": total_time,
        "max_mem_mb": max_mem_mb,
    }

    print(f"[{cfg.label}] total time: {total_time:.1f} s")
    if max_mem_mb is not None:
        print(f"[{cfg.label}] peak GPU memory: {max_mem_mb:.1f} MB")

    return model, logs