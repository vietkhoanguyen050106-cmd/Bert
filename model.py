from transformers import AutoModelForSequenceClassification


MODEL_NAME = "distilroberta-base"
NUM_LABELS = 2


def build_student():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    return model