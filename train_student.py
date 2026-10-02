from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from Bert.dataset import load_sst2
from Bert.model_student import build_student
from Bert.config import STUDENT_BASELINE_DIR
from Bert.utils import compute_metrics, make_training_args, report


OUTPUT_DIR = "./student_baseline"

def main():    
    dataset, tokenizer = load_sst2()


    student = build_student()


    data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


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


trainer = Trainer(
    model=student,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator
)


trainer.train()


trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

report("student_baseline", trainer, student)
 
if __name__ == "__main__":
    main()