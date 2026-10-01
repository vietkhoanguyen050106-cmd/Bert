from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from dataset import load_sst2
from model_student import build_student


OUTPUT_DIR = "./student_baseline"


# 1. Load SST-2 and BERT tokenizer
dataset, tokenizer = load_sst2()

# 2. Build 6-layer BERT Student
student = build_student()

# 3. Dynamic padding
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# 4. Training configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=student,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator
)

# 6. Train Student
trainer.train()

# 7. Save Student
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)