import glob
import json
import os
import re
 
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from Bert.config import OUTPUT_ROOT, TEACHER_DIR
 
 
def load_curve(model_dir):
    
    path = os.path.join(model_dir, "log_history.json")
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        log = json.load(f)
    points = [(e["epoch"], e["eval_accuracy"] * 100) for e in log if "eval_accuracy" in e]
    return [p[0] for p in points], [p[1] for p in points]
 
 
def main():
    sizes = sorted(
        int(re.search(r"_(\d+)L$", d).group(1))
        for d in glob.glob(os.path.join(OUTPUT_ROOT, "student_baseline_*L"))
    )
    if not sizes:
        raise SystemExit("No trained student found in ./outputs")
 
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), squeeze=False, sharey=True)
    teacher_x, teacher_y = load_curve(TEACHER_DIR)
 
    for ax, layers in zip(axes[0], sizes):
        ax.plot(teacher_x, teacher_y, "k--", label="teacher (12L)")
        for mode, color in [("baseline", "tab:orange"), ("distill", "tab:blue")]:
            x, y = load_curve(os.path.join(OUTPUT_ROOT, f"student_{mode}_{layers}L"))
            ax.plot(x, y, color=color, marker="o", markersize=3, label=f"student {mode} ({layers}L)")
        ax.set_title(f"{layers}-layer student")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("SST-2 validation accuracy (%)")
 
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, "learning_curves.png"), dpi=150)
    print("saved", os.path.join(OUTPUT_ROOT, "learning_curves.png"))
 
 
if __name__ == "__main__":
    main()
 