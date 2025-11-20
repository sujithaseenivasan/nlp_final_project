# train.py
import argparse, os, random
import torch
import torch.nn as nn
from tqdm import tqdm
from model import MiniTransformerLM

def set_seed(seed=1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def evaluate(model, X_val, Y_val, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for xb, yb in batch_iter(X_val, Y_val, batch_size, shuffle=False, device=device):
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            total_loss += float(loss) * yb.numel()
            total_tokens += yb.numel()
    ppl = torch.exp(torch.tensor(total_loss / max(total_tokens,1)))
    return float(ppl)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="tiny_corpus.txt")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--attn_dropout", type=float, default=0.0)
    ap.add_argument("--resid_dropout", type=float, default=0.1)
    ap.add_argument("--disable_positional", action="store_true")
    ap.add_argument("--attn_dropout_ablate", type=float, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", type=str, default="checkpoint.pt")
    args = ap.parse_args()

    set_seed(1337)

    lines = load_lines(args.data)
    stoi, itos = build_vocab(lines, min_freq=1)
    X, Y = make_dataset(lines, stoi, seq_len=args.seq_len)
    n = int(0.9 * X.size(0))
    X_tr, Y_tr = X[:n], Y[:n]
    X_val, Y_val = X[n:], Y[n:]

    model = MiniTransformerLM(
        vocab_size=len(itos),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.seq_len,
        attn_dropout=args.attn_dropout_ablate or args.attn_dropout,
        resid_dropout=args.resid_dropout,
        disable_positional=args.disable_positional
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"Vocab={len(itos)} | Train seqs={X_tr.size(0)} | Val seqs={X_val.size(0)} | Device={args.device}")

    best_ppl = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(batch_iter(X_tr, Y_tr, args.batch_size, shuffle=True, device=args.device),
                    total=(X_tr.size(0) + args.batch_size - 1)//args.batch_size,
                    desc=f"Epoch {ep}")
        for xb, yb in pbar:
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss))

        val_ppl = evaluate(model, X_val, Y_val, args.batch_size, args.device)
        print(f"Epoch {ep}: val ppl = {val_ppl:.2f}")
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos, "args": vars(args)}, args.save)

    # --- Attention visualization on a fixed sentence ---
    test_sentence = "the quick brown fox jumps over the lazy dog"
    toks = tokenize(test_sentence)
    ids = [stoi.get(t, stoi["<unk>"]) for t in toks]
    # pad/trim to seq_len
    ids = ids[:args.seq_len]
    while len(ids) < args.seq_len:
        ids.append(0)

    inp = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits, attn = model(inp, return_attn=True)  # (B,H,T,T)

    fig = plot_attention(tokens=toks + ["<pad>"]*(args.seq_len-len(toks)), attn=attn, head=0,
                         title="Self-Attention (baseline)")
    fig.savefig("attention_baseline.png", dpi=160)
    print("Saved attention heatmap to attention_baseline.png")


if __name__ == "__main__":
    main()
