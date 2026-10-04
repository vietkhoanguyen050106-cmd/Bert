from transformers import DataCollatorWithPadding, Trainer
 
from Bert.config import TEACHER_DIR
from Bert.dataset import load_sst2
from Bert.model_teacher import build_teacher
from Bert.utils import compute_metrics, count_parameters, make_training_args, save_history
 
 
def main():
    
    dataset, tokenizer = load_sst2()
 
   
    teacher = build_teacher()
 
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
    
    trainer = Trainer(
        model=teacher,
        args=make_training_args(TEACHER_DIR),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
 
    
    trainer.train()
    trainer.save_model(TEACHER_DIR)
    tokenizer.save_pretrained(TEACHER_DIR)
    save_history(trainer, TEACHER_DIR)
 
    
    metrics = trainer.evaluate()
    print(f"[teacher] accuracy={metrics['eval_accuracy']:.4f} "
          f"params={count_parameters(teacher) / 1e6:.1f}M")
 
 
if __name__ == "__main__":
    main()