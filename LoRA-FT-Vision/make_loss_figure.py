"""Render the training loss curve from the TensorBoard logs written by train.py.

Equivalent to `tensorboard --logdir qwen2.5-vl-3b-chartqa-lora`, but produces a
static PNG (a reproducible figure) by reading the event files directly.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGDIR = "qwen2.5-vl-3b-chartqa-lora"

# Find the most recent event file under the run dir.
event_files = sorted(
    glob.glob(os.path.join(LOGDIR, "**", "events.out.tfevents*"), recursive=True),
    key=os.path.getmtime,
)
assert event_files, f"no TensorBoard event files under {LOGDIR}"
event_path = event_files[-1]
print("reading", event_path)

acc = EventAccumulator(event_path, size_guidance={"scalars": 0})
acc.Reload()
print("available scalar tags:", acc.Tags()["scalars"])


def series(tag):
    evs = acc.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]


fig, ax = plt.subplots(figsize=(8, 5))

loss_steps, loss_vals = series("train/loss")
ax.plot(loss_steps, loss_vals, "-o", color="#1f77b4", label="train/loss", lw=2, ms=5)
ax.set_xlabel("step")
ax.set_ylabel("training loss")
ax.set_title("ChartQA LoRA fine-tune — training loss\n"
             "(Qwen2.5-VL-3B, 4-bit QLoRA, smoke run: 60 steps)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

# Annotate first/last loss so the drop is legible.
ax.annotate(f"{loss_vals[0]:.3f}", (loss_steps[0], loss_vals[0]),
            textcoords="offset points", xytext=(6, 8), fontsize=9)
ax.annotate(f"{loss_vals[-1]:.3f}", (loss_steps[-1], loss_vals[-1]),
            textcoords="offset points", xytext=(-10, 10), fontsize=9)

# If token accuracy was logged, overlay it on a twin axis.
if "train/mean_token_accuracy" in acc.Tags()["scalars"]:
    acc_steps, acc_vals = series("train/mean_token_accuracy")
    ax2 = ax.twinx()
    ax2.plot(acc_steps, acc_vals, "--s", color="#2ca02c",
             label="train/mean_token_accuracy", lw=1.5, ms=4, alpha=0.8)
    ax2.set_ylabel("mean token accuracy", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="center right")

fig.tight_layout()
os.makedirs("assets", exist_ok=True)
out = "assets/loss_curve.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
print(f"loss: {loss_vals[0]:.3f} -> {loss_vals[-1]:.3f}  over {len(loss_vals)} logged points")
