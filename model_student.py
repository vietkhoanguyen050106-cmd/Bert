import torch
from transformers import AutoModelForSequenceClassification
from Bert.config import MODEL_NAME, NUM_LABELS, NUM_STUDENT_LAYERS


MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
NUM_STUDENT_LAYERS = 6


def build_student():
    student = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    student.bert.encoder.layer = torch.nn.ModuleList(
        student.bert.encoder.layer[:NUM_STUDENT_LAYERS]
    )

    student.config.num_hidden_layers = NUM_STUDENT_LAYERS

    return student