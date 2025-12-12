import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

from model import MiniTransformerLM, TrainConfig, train_one_config


def plot_mini_transformer_attention_heatmap(model, tokenizer, text,
                                            layer_idx=0, head_idx=None,
                                            max_len=50):
    """
    Run the mini Transformer on `text` and show an attention heatmap
    for a given layer and head (or averaged over heads).
    """
    model.eval()
    device = next(model.parameters()).device

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"].to(device)

    # turn on attention saving for the chosen layer
    block = model.blocks[layer_idx].self_attn
    block.save_attn = True
    with torch.no_grad():
        _ = model(input_ids)
    block.save_attn = False

    attn = block.last_attn_weights  # (B, n_heads, T, T)
    if attn is None:
        raise RuntimeError(
            "No attention weights were saved. "
        )

    # choose head or average over heads
    if head_idx is None:
        attn_map = attn.mean(dim=1)[0].numpy()      # (T, T)
        title_head = "avg heads"
    else:
        attn_map = attn[0, head_idx].numpy()        # (T, T)
        title_head = f"head {head_idx}"

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [t.replace("Ġ", "") for t in tokens] 


    # plot heatmap
    plt.figure(figsize=(0.5 * len(tokens), 0.5 * len(tokens)))
    plt.imshow(attn_map)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("Key (attended to)")
    plt.ylabel("Query (attending from)")
    plt.title(f"MiniTransformer attention - layer {layer_idx}, {title_head}")
    plt.colorbar(label="Attention weight")
    plt.tight_layout()

    plt.savefig("mini_transformer_attention_heatmap.png", dpi=200, bbox_inches="tight")
    print(f"Saved attention heatmap to: mini_transformer_attention_heatmap.png")

    plt.show()


def run_attention_experiment():
    """
    Train a small rotary MiniTransformer on TinyStories and
    visualize attention on a small example.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TrainConfig(
        epochs=2,
        n_train=5000,
        n_val=1000,
        max_len=128,
        batch_size=16,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.0,
        pos_encoding_type="rotary",
        label="rotary_attention_demo",
    )

    model, logs = train_one_config(cfg, device=device)
    text = "I thought the movie would be boring, but it turned out to be amazing."

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    plot_mini_transformer_attention_heatmap(
        model,
        tokenizer,
        text,
        layer_idx=1,
        head_idx=None,
        max_len=50, 
    )


if __name__ == "__main__":
    run_attention_experiment()