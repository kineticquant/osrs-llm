from datasets import load_from_disk

dataset = load_from_disk("data/clean")

print("Confirming loaded binary training file(s) in data/clean to confirm data pipeline extract.")
print(f"Total examples: {len(dataset)}")
print("\nFirst 5 entries:")
for i in range(5):
    print(f"\n--- Example {i+1} ---")
    print(dataset[i]["text"][:1000])  # First 1k chars

print("\nLast 5 entries:")
for i in range(-5, 0):
    print(f"\n--- Example {i+1} ---")
    print(dataset[i]["text"][:1000])