import argparse
 
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
 
from Bert.config import ALPHA, BETA, NUM_EPOCHS, TEACHER_DIR, TEMPERATURE, student_dir
from Bert.dataset import load_sst2
from Bert.model_student import build_student
from Bert.model_teacher import build_teacher
from Bert.utils import compute_metrics, count_parameters, make_training_args
 
 
class DistillationTrainer(Trainer):
 
    def __init__(self, *args, teacher, temperature, alpha, beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.to(self.args.device)
        self.teacher.eval()                        
        for p in self.teacher.parameters():
            p.requires_grad = False               
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        use_hidden = self.beta > 0
 
       
        student_out = model(**inputs, output_hidden_states=use_hidden)
        ce_loss = student_out.loss
 
        
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            teacher_out = self.teacher(**teacher_inputs, output_hidden_states=use_hidden)
 
       
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_out.logits / T, dim=-1),
            F.softmax(teacher_out.logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
 
        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
 
       
        if use_hidden:
            mask = inputs["attention_mask"].bool()
            s_hidden = student_out.hidden_states[-1][mask]
            t_hidden = teacher_out.hidden_states[-1][mask]
            target = torch.ones(s_hidden.size(0), device=s_hidden.device)
            loss = loss + self.beta * F.cosine_embedding_loss(s_hidden, t_hidden, target)
 
        if return_outputs:
            
            return loss, SequenceClassifierOutput(loss=loss, logits=student_out.logits)
        return loss
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--init", default="alternate", choices=["alternate", "first"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--teacher_dir", default=TEACHER_DIR)
    args = parser.parse_args()
    output_dir = student_dir("distill", args.layers)
 
    dataset, tokenizer = load_sst2()
    teacher = build_teacher(args.teacher_dir)       
    student = build_student(args.layers, args.init)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
    trainer = DistillationTrainer(
        model=student,
        args=make_training_args(output_dir, epochs=args.epochs),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
    )
 
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
 
    metrics = trainer.evaluate()
    print(f"[student_distill_{args.layers}L] accuracy={metrics['eval_accuracy']:.4f} "
          f"params={count_parameters(student) / 1e6:.1f}M")
 
 
if __name__ == "__main__":
    main()
 