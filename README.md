# Knowledge Distillation: How Small Can a Neural Network Become?

Teacher and students are trained on SST-2 (GLUE, binary sentiment) with Hugging Face
Transformers. We study the trade-off between **model size**, **computational
efficiency** and **prediction performance**.

## 1. Experimental design

| Model | Layers | Trained how | Role |
|---|---|---|---|
| Teacher | 12 (BERT-base, ~110M) | fine-tuned on SST-2 | upper bound, source of soft targets |
| Student baseline (2/4/6 L) | 2, 4, 6 | hard labels only | control experiment |
| Student distilled (2/4/6 L) | 2, 4, 6 | hard labels + teacher | the method under study |

* Everything uses the same data, epochs, learning rate, seed and initialisation scheme,
  so differences come only from model size and from distillation.
* Gap `distilled - baseline` = benefit of distillation.
  Gap `teacher - distilled` = what is still lost by making the model smaller.
* Scripts are run from the repository root (the folder that contains `Bert/`).

## 2. Run on Kaggle (GPU + Internet on)

```python
!git clone https://github.com/<user>/<repo>.git
%cd <repo>
!pip install -q transformers datasets evaluate accelerate

# whole study (teacher, 3 student sizes x 2 modes, benchmark):
!python -m Bert.run_all --layers 2 4 6

# or step by step:
!python -m Bert.train_teacher
!python -m Bert.train_student --layers 6
!python -m Bert.train_distill --layers 6
!python -m Bert.benchmark
```

Results: `outputs/comparison.csv`, `comparison.json`, `comparison.png`.
Zip them before the session ends: `!zip -r outputs.zip outputs`.

## 3. Files

| File | What it does |
|---|---|
| `config.py` | all constants and output paths |
| `dataset.py` | `load_sst2()` -> tokenized dataset + tokenizer |
| `model_teacher.py` | `build_teacher(path=None)` |
| `model_student.py` | `build_student(num_layers, init)` |
| `utils.py` | `compute_metrics`, `make_training_args`, `count_parameters` |
| `train_teacher.py` | step 1, fine-tune the teacher |
| `train_student.py` | step 2, baseline student |
| `train_distill.py` | step 3, `DistillationTrainer` |
| `benchmark.py` | accuracy, agreement, size, latency table and plot |
| `run_all.py` | runs everything, skips finished models |

## 4. Why these functions and not others

| Choice | Alternative | Reason |
|---|---|---|
| `AutoModelForSequenceClassification` | `BertModel` + hand-written head | already contains the head, computes cross-entropy from labels, saves/loads in one call |
| `Trainer` | own PyTorch loop | fp16, evaluation, checkpoints, best-model selection for free; only `compute_loss` is overridden |
| `DataCollatorWithPadding` | `padding="max_length"` | SST-2 sentences are short; per-batch padding saves compute with identical results |
| `bert-base-uncased` for both | DistilBERT as student | the study needs a student whose depth we control; same hidden size also allows hidden-state alignment |
| Student built from pretrained layers | random initialisation | starts with language knowledge, converges faster, ends higher |
| `alternate` layer selection | first-k layers | keeps low- and high-level layers (DistilBERT recipe); `--init first` is available to compare |
| KL divergence with temperature | MSE on logits, hard labels only | compares full probability distributions; soft targets carry information a 0/1 label does not |
| `T^2` factor | none | keeps gradient size comparable to cross-entropy when T > 1 |
| `alpha` mixing | teacher loss only | the true label protects the student from teacher mistakes |
| cosine loss on last hidden state (`--beta`) | none | aligns internal representations; DistilBERT's third loss; optional |
| `teacher.eval()`, `requires_grad=False`, `torch.no_grad()` | leave the teacher trainable | teacher stays fixed and uses no gradient memory |
| validation set for reporting | test set | GLUE test labels are not public |
| agreement + latency + size in `benchmark.py` | accuracy only | the project question is a trade-off, so all three axes are measured |

## 5. Getting the student as close to the teacher as possible

Change one setting at a time and compare validation accuracy and
`agreement_with_teacher` in the benchmark table:

1. `--temperature` in {1, 2, 4}
2. `--alpha` in {0.3, 0.5, 0.7}
3. `--beta` in {0, 1} (hidden-state loss)
4. `--init` in {alternate, first}
5. `--epochs` in {3, 5}

## 6. Notes and limitations

* One run per setting; SST-2 validation has only 872 sentences, so differences below
  about 0.5 points are within noise. Repeat with different seeds (`SEED` in
  `config.py`) before drawing strong conclusions.
* Latency depends on the GPU, so report ratios (`speedup_x`), not absolute values.
* The code was syntax-checked but not executed end-to-end in the authoring
  environment (no GPU or Hugging Face access); run it once on Kaggle and fix any
  environment-specific error that appears.
