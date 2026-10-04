from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


def load_sst2():
     
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        raw_dataset = load_dataset("glue", "sst2")


        def tokenize_batch(batch):
            return tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=MAX_LENGTH
            )
        
        return (
        raw_dataset.map(
            tokenize_batch, batched=True, remove_columns=["sentence", "idx"]
        ),
        tokenizer,
    )