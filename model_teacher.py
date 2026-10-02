from transformers import AutoModelForSequenceClassification
from Bert.config import MODEL_NAME, NUM_LABELS

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2


def build_teacher():
    teacher = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    return teacher