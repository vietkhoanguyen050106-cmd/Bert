from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


def load_sst2():
    
    dataset = load_dataset("nyu-mll/glue", "sst2")


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    def tokenize_function(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )

    
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["sentence", "idx"]
    )

    
    tokenized_dataset = tokenized_dataset.rename_column(
        "label",
        "labels"
    )

    return tokenized_dataset, tokenizer