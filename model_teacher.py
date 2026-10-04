from transformers import AutoModelForSequenceClassification
from Bert.config import MODEL_NAME, NUM_LABELS

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2


def build_teacher(path=None):
    teacher = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    return  AutoModelForSequenceClassification.from_pretrained(
        path or MODEL_NAME, num_labels=NUM_LABELS
    )