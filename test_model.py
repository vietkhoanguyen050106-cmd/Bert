from Bert.dataset import load_sst2
from Bert.model import build_student


# 1. Load dataset and tokenizer
dataset, tokenizer = load_sst2()

# 2. Build Student model
student = build_student()

# 3. Get one example
example = dataset["train"][0]

print("Example:")
print(example)

print("\nStudent model:")
print(student)