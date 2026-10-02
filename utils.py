import json
import os

import evaluate
import numpy as np
import torch
from transformers import TrainingArguments

from Bert.config import SEED

_accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return _accuracy.compute(predictions=predictions, references=labels)


def make_training_args(output_dir, epochs=3, lr=2e-5, batch_size=32):
   
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )


def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters())


def report(name, trainer, model, results_file="./outputs/results.json"):
    
    accuracy = trainer.evaluate()["eval_accuracy"]
    params = count_parameters(model)
    print(f"\n[{name}] accuracy = {accuracy:.4f} | tham số = {params / 1e6:.1f}M")

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    results[name] = {"accuracy": accuracy, "params_million": round(params / 1e6, 1)}
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)