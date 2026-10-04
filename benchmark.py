import csv
import glob
import json
import os
import time
 
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
 
from Bert.config import OUTPUT_ROOT, TEACHER_DIR
from Bert.dataset import load_sst2
from Bert.utils import count_parameters
 
 
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
 
 
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        predictions.append(model(**batch).logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)
 
 
@torch.no_grad()
def measure_latency(model, tokenizer, device, runs=100, warmup=10):
    """Average milliseconds to classify ONE sentence (batch size 1, 64 tokens)."""
    model.eval()
    sample = tokenizer(
        "a gorgeous, witty and moving film",
        padding="max_length", max_length=64, return_tensors="pt",
    ).to(device)
    for _ in range(warmup):                 
        model(**sample)
    sync(device)
    start = time.perf_counter()
    for _ in range(runs):
        model(**sample)
    sync(device)
    return (time.perf_counter() - start) / runs * 1000
 
 
def evaluate_model(model_dir, loader, labels, tokenizer, device, teacher_preds=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    params = count_parameters(model)
 
    sync(device)
    start = time.perf_counter()
    predictions = predict(model, loader, device)      
    sync(device)
    elapsed = time.perf_counter() - start
 
    row = {
        "model": os.path.basename(model_dir),
        "layers": model.config.num_hidden_layers,
        "params_M": round(params / 1e6, 1),
        "size_MB": round(params * 4 / 1e6, 1),         
        "accuracy": round((predictions == labels).float().mean().item() * 100, 2),
        "latency_ms": round(measure_latency(model, tokenizer, device), 2),
        "throughput_sent_per_s": round(len(labels) / elapsed, 1),
    }
    if teacher_preds is not None:
        row["agreement_with_teacher"] = round(
            (predictions == teacher_preds).float().mean().item() * 100, 2
        )
    return row, predictions
 
 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, tokenizer = load_sst2()
    validation = dataset["validation"]
    labels = torch.tensor(validation["label"])
    loader = DataLoader(
        validation, batch_size=64,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
 
   
    teacher_row, teacher_preds = evaluate_model(
        TEACHER_DIR, loader, labels, tokenizer, device
    )
    teacher_row["agreement_with_teacher"] = 100.0
    rows = [teacher_row]
 
    student_dirs = sorted(
        d for d in glob.glob(os.path.join(OUTPUT_ROOT, "student_*"))
        if os.path.exists(os.path.join(d, "config.json"))
    )
    for directory in student_dirs:
        row, _ = evaluate_model(directory, loader, labels, tokenizer, device, teacher_preds)
        rows.append(row)
 
    for row in rows:
        row["accuracy_gap_vs_teacher"] = round(row["accuracy"] - teacher_row["accuracy"], 2)
        row["compression_x"] = round(teacher_row["params_M"] / row["params_M"], 2)
        row["speedup_x"] = round(teacher_row["latency_ms"] / row["latency_ms"], 2)
 
    # ---- print table ----
    columns = ["model", "layers", "params_M", "size_MB", "accuracy",
               "accuracy_gap_vs_teacher", "agreement_with_teacher",
               "latency_ms", "speedup_x", "compression_x"]
    print("\n" + " | ".join(columns))
    for row in rows:
        print(" | ".join(str(row.get(c, "")) for c in columns))
 
    # ---- save json / csv ----
    with open(os.path.join(OUTPUT_ROOT, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    all_columns = list(rows[0].keys()) + ["agreement_with_teacher"]
    all_columns = list(dict.fromkeys(all_columns))
    with open(os.path.join(OUTPUT_ROOT, "comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
 
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        plt.figure(figsize=(7, 4.5))
        for row in rows:
            marker = "*" if row["model"] == "teacher" else ("o" if "distill" in row["model"] else "s")
            plt.scatter(row["params_M"], row["accuracy"], marker=marker, s=90)
            plt.annotate(row["model"], (row["params_M"], row["accuracy"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
        plt.xlabel("Parameters (millions)")
        plt.ylabel("SST-2 validation accuracy (%)")
        plt.title("Model size vs accuracy")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, "comparison.png"), dpi=150)
    except Exception as error:                      
        print("plot skipped:", error)
 
 
if __name__ == "__main__":
    main()