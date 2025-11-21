from datasets import load_dataset
import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer

def tokenize_data(max_len=128, n_train=20000, n_val=5000):
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
        B, T = input_ids.shape
        x = self.embed(input_ids)

        if not self.disable_positional:
            pe = sinusoidal_position_encoding(T, self.d_model, input_ids.device)
            x = x + pe.unsqueeze(0)

        mask = causal_tgt_mask(T, input_ids.device)
        memory = torch.zeros(B, 1, self.d_model, device=input_ids.device)

        h = self.decoder(tgt=x, memory=memory, tgt_mask=mask)
        h = self.ln_f(h)
        return self.head(h)
    

def get_batch(ds, batch_size, device):
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i+batch_size]
        yield torch.tensor(batch["input_ids"], device=device)


def train():
    max_len = 128
    batch_size = 16
    epochs = 1
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, train_ds, val_ds = tokenize_data(max_len)
    model = MiniTransformerLM(vocab_size=tokenizer.vocab_size, max_len=max_len).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for input_ids in get_batch(train_ds, batch_size, device):
            x = input_ids[:, :-1]
            y = input_ids[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for input_ids in get_batch(val_ds, batch_size, device):
                x = input_ids[:, :-1]
                y = input_ids[:, 1:]
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                val_loss += loss.item()
                n += 1

        print("validation loss:", val_loss / max(n, 1))


if __name__ == "__main__":
    train()
    
