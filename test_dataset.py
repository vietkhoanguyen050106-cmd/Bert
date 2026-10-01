from dataset import load_sst2


dataset, tokenizer = load_sst2()

print(dataset)

print("\nTrain size:")
print(len(dataset["train"]))

print("\nValidation size:")
print(len(dataset["validation"]))

print("\nTest size:")
print(len(dataset["test"]))

print("\nFirst example:")
print(dataset["train"][0])

print("\nTokenizer:")
print(tokenizer)