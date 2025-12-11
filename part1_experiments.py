import math
import torch
import matplotlib.pyplot as plt

from model import TrainConfig, train_one_config



# Helper functions for plotting results
def plot_experiment(logs_list, labels, title_suffix=""):
    """
    Plot training and validation loss curves for a list of experiments.
    """
    # Training loss vs step
    plt.figure()
    for logs, label in zip(logs_list, labels):
        plt.plot(logs["train_loss_history"], label=f"{label} (train)")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Train loss " + title_suffix)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Validation loss vs epoch
    plt.figure()
    for logs, label in zip(logs_list, labels):
        plt.plot(logs["val_loss_history"], marker="o", label=f"{label} (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Val loss")
    plt.title("Validation loss " + title_suffix)
    plt.legend()
    plt.grid(True)
    plt.show()


# Helper functions for running experiments
def run_positional_experiments():
    """
    Compare sinusoidal, learned, rotary, and no positional encodings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        epochs=2,
        n_train=10000,
        n_val=2000,
        max_len=128,
        batch_size=16,
    )

    cfg_sin = TrainConfig(
        pos_encoding_type="sinusoidal",
        label="sinusoidal_pe",
        **common_kwargs,
    )
    cfg_learned = TrainConfig(
        pos_encoding_type="learned",
        label="learned_pe",
        **common_kwargs,
    )
    cfg_rotary = TrainConfig(
        pos_encoding_type="rotary",
        label="rotary_pe",
        **common_kwargs,
    )
    cfg_none = TrainConfig(
        pos_encoding_type="none",
        label="no_pe",
        **common_kwargs,
    )

    _, logs_sin = train_one_config(cfg_sin, device=device)
    _, logs_learned = train_one_config(cfg_learned, device=device)
    _, logs_rotary = train_one_config(cfg_rotary, device=device)
    _, logs_none = train_one_config(cfg_none, device=device)

    return logs_sin, logs_learned, logs_rotary, logs_none


def run_size_experiments():
    """
    Compare a small vs larger model to study size vs efficiency trade-offs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        epochs=2,
        n_train=10000,
        n_val=2000,
        max_len=128,
        batch_size=16,
        pos_encoding_type="sinusoidal",  
    )

    cfg_small = TrainConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        label="small_model",
        **common_kwargs,
    )
    cfg_big = TrainConfig(
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=512,
        label="big_model",
        **common_kwargs,
    )

    _, logs_small = train_one_config(cfg_small, device=device)
    _, logs_big = train_one_config(cfg_big, device=device)

    return logs_small, logs_big

if __name__ == "__main__":
    # Positional encoding experiments
    logs_sin, logs_learned, logs_rotary, logs_none = run_positional_experiments()
    plot_experiment(
        [logs_sin, logs_learned, logs_rotary, logs_none],
        ["sin", "learned", "rotary", "none"],
        title_suffix="(positional encodings)",
    )

    # Model size experiments
    logs_small, logs_big = run_size_experiments()
    plot_experiment(
        [logs_small, logs_big],
        ["small", "big"],
        title_suffix="(model size)",
    )
