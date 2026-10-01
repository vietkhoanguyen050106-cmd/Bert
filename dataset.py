from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


def load_sst2():
    # Load SST-2 dataset from Hugging Face
    dataset = load_dataset("nyu-mll/glue", "sst2")

    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization function
    def tokenize_function(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )

    # Remove unnecessary columns
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["sentence", "idx"]
    )

    # Rename label to the name expected by the model
    tokenized_dataset = tokenized_dataset.rename_column(
        "label",
        "labels"
    )

    return tokenized_dataset, tokenizer