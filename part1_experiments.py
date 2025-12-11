import math
import torch
import matplotlib.pyplot as plt

from model import TrainConfig, train_one_config



# Helper functions for plotting results
def plot_experiment(logs_list, labels, title_suffix="", save_prefix="result"):
    """
    Plot training and validation loss curves for a list of experiments.
    """
    # Training loss vs step
    plt.figure(figsize=(8, 5))
    for logs, label in zip(logs_list, labels):
        plt.plot(logs["train_loss_history"], label=f"{label} (train)")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Train loss " + title_suffix)
    plt.legend()
    plt.grid(True)
    
    train_filename = f"{save_prefix}_train_loss.png"
    plt.savefig(train_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {train_filename}")

    plt.show()

    # Validation loss vs epoch
    plt.figure(figsize=(8, 5))
    for logs, label in zip(logs_list, labels):
        plt.plot(logs["val_loss_history"], marker="o", label=f"{label} (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Val loss")
    plt.title("Validation loss " + title_suffix)
    plt.legend()
    plt.grid(True)
    
    val_filename = f"{save_prefix}_val_loss.png"
    plt.savefig(val_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {val_filename}")

    plt.show()

def save_summary_table_png(logs_list, labels, filename="summary_table.png", title=None):
    """
    Create a compact table (as a PNG image) summarizing experiments.

    Columns:
      - Label
      - Val loss
      - Val perplexity
      - Time (s)
      - Max memory (MB)

    The table is designed to be narrow enough to fit nicely in a report.
    """
    import math

    # Collect rows
    table_data = []
    headers = ["Label", "VaLoss", "VaPPL", "Time (s)", "Mem (MB)"]

    for logs, label in zip(logs_list, labels):
        val_losses = logs["val_loss_history"]
        final_val_loss = val_losses[-1]
        val_ppl = math.exp(final_val_loss)

        total_time = logs.get("total_time_sec", float("nan"))
        max_mem = logs.get("max_mem_mb", float("nan"))

        row = [
            label,
            f"{final_val_loss:.3f}",
            f"{val_ppl:.1f}",
            f"{total_time:.1f}",
            f"{max_mem:.0f}" if max_mem == max_mem else "nan",
        ]
        table_data.append(row)

    # Make figure
    fig, ax = plt.subplots(figsize=(4.0, 0.5 + 0.3 * len(table_data)))
    ax.axis("off")

    if title is not None:
        ax.set_title(title, fontsize=10, pad=6)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)  # slightly taller for readability

    # Adjust column widths to keep table relatively skinny
    col_widths = [0.18, 0.16, 0.18, 0.18, 0.18]
    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved summary table image to: {filename}")
    plt.close(fig)

def summarize_experiments(logs_list, labels):
    """
    Print summary table for experiments:
      - final train loss
      - final val loss
      - train/val perplexity
      - total time
      - peak GPU memory
      - average step time
    """
    import math

    print("\n=== Experiment Summary ===")
    header = (
        f"{'Label':<10} {'TrLoss':>8} {'VaLoss':>8} "
        f"{'TrPPL':>8} {'VaPPL':>8} "
        f"{'Time':>8} {'AvgStep':>8} {'MemMB':>8}"
    )
    print(header)
    print("-" * len(header))

    for logs, label in zip(logs_list, labels):
        train_losses = logs["train_loss_history"]
        val_losses = logs["val_loss_history"]
        step_times = logs["step_times"]

        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]

        train_ppl = math.exp(final_train_loss)
        val_ppl = math.exp(final_val_loss)

        total_time = logs.get("total_time_sec", float("nan"))
        max_mem = logs.get("max_mem_mb", float("nan"))
        avg_step_time = sum(step_times) / len(step_times)

        print(
            f"{label:<10} "
            f"{final_train_loss:>8.3f} {final_val_loss:>8.3f} "
            f"{train_ppl:>8.1f} {val_ppl:>8.1f} "
            f"{total_time:>8.1f} {avg_step_time:>8.3f} {max_mem:>8.0f}"
        )


# Helper functions for running experiments
def run_positional_experiments():
    """
    Compare sinusoidal, learned, rotary, and no positional encodings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        epochs=4,
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
        pos_encoding_type="rotary", # shown to be best from pe experiment
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

def run_dropout_experiments():
    """
    Compare different dropout magnitudes to study overfitting and performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        epochs=2,
        n_train=10000,
        n_val=2000,
        max_len=128,
        batch_size=16,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        pos_encoding_type="rotary", # shown to be best from pe experiment
        lr=3e-4,
    )

    cfg_do0 = TrainConfig(
        dropout=0.0,
        label="dropout_0.0",
        **common_kwargs,
    )
    cfg_do01 = TrainConfig(
        dropout=0.1,
        label="dropout_0.1",
        **common_kwargs,
    )
    cfg_do03 = TrainConfig(
        dropout=0.3,
        label="dropout_0.3",
        **common_kwargs,
    )

    _, logs_do0 = train_one_config(cfg_do0, device=device)
    _, logs_do01 = train_one_config(cfg_do01, device=device)
    _, logs_do03 = train_one_config(cfg_do03, device=device)

    return logs_do0, logs_do01, logs_do03

if __name__ == "__main__":
    # Positional encoding experiments
    logs_sin, logs_learned, logs_rotary, logs_none = run_positional_experiments()
    logs_list = [logs_sin, logs_learned, logs_rotary, logs_none]
    pe_labels = ["sin", "learned", "rotary", "none"]
    plot_experiment(
        logs_list,
        pe_labels,
        title_suffix="(positional encodings)",
        save_prefix="positional_encodings"
    )
    summarize_experiments(logs_list, pe_labels)
    save_summary_table_png(
        logs_list,
        pe_labels,
        filename="positional_encodings_table.png",
        title="Positional Encoding Ablation"
    )

    # Model size experiments
    logs_small, logs_big = run_size_experiments()
    logs_list = [logs_small, logs_big]
    ms_labels = ["small", "big"]
    plot_experiment(
        logs_list,
        ms_labels,
        title_suffix="(model size)",
        save_prefix="model_size"
    )
    summarize_experiments(logs_list, ms_labels)
    save_summary_table_png(
        logs_list,
        ms_labels,
        filename="model_size_table.png",
        title="Model Size Ablation"
    )

    # Dropout experiments
    logs_do0, logs_do01, logs_do03 = run_dropout_experiments()
    logs_list = [logs_do0, logs_do01, logs_do03]
    do_labels = ["do=0.0", "do=0.1", "do=0.3"]
    plot_experiment(
        logs_list,
        do_labels,
        title_suffix="(dropout ablation)",
        save_prefix="dropout_ablation"
    )
    summarize_experiments(logs_list, do_labels)
    save_summary_table_png(
        logs_list,
        do_labels,
        filename="dropout_ablation_table.png",
        title="Dropout Magnitude Ablation"
    )
