import argparse
 
from transformers import DataCollatorWithPadding, Trainer
 
from Bert.config import NUM_EPOCHS, student_dir
from Bert.dataset import load_sst2
from Bert.model_student import build_student
from Bert.utils import compute_metrics, count_parameters, make_training_args
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=6, help="encoder layers in the student")
    parser.add_argument("--init", default="alternate", choices=["alternate", "first"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    output_dir = student_dir("baseline", args.layers)
 
    dataset, tokenizer = load_sst2()
    student = build_student(args.layers, args.init)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
    trainer = Trainer(
        model=student,
        args=make_training_args(output_dir, epochs=args.epochs),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
 
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
 
    metrics = trainer.evaluate()
    print(f"[student_baseline_{args.layers}L] accuracy={metrics['eval_accuracy']:.4f} "
          f"params={count_parameters(student) / 1e6:.1f}M")
 
 
if __name__ == "__main__":
    main()