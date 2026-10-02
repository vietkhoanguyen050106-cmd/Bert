import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, Trainer
 
from Bert.config import ALPHA, STUDENT_DISTILL_DIR, TEACHER_DIR, TEMPERATURE
from Bert.dataset import load_sst2
from Bert.model_student import build_student
from Bert.model_teacher import build_teacher
from Bert.utils import compute_metrics, make_training_args, report
 
 
class DistillationTrainer(Trainer):

 
    def __init__(self, *args, teacher, temperature, alpha, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.to(self.args.device)  
        self.teacher.eval()                          
        for p in self.teacher.parameters():
            p.requires_grad = False                  
        self.temperature = temperature
        self.alpha = alpha
 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        student_out = model(**inputs)
        ce_loss = student_out.loss
 
       
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            teacher_logits = self.teacher(**teacher_inputs).logits
 
        
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_out.logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
 
        
        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return (loss, student_out) if return_outputs else loss
 
 
def main():
    
    dataset, tokenizer = load_sst2()
 
    
    teacher = build_teacher(TEACHER_DIR)
    student = build_student()
 
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
    
    trainer = DistillationTrainer(
        model=student,
        args=make_training_args(STUDENT_DISTILL_DIR),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        teacher=teacher,
        temperature=TEMPERATURE,
        alpha=ALPHA,
    )
 
    
    trainer.train()
    trainer.save_model(STUDENT_DISTILL_DIR)
    tokenizer.save_pretrained(STUDENT_DISTILL_DIR)
    report("student_distilled", trainer, student)
 
 
if __name__ == "__main__":
    main()