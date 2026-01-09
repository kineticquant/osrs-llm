import os
from datasets import load_from_disk


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print(f"Project Root: {project_root}")
print(f"Script Directory: {script_dir}")



if script_dir == project_root:
    dataset = load_from_disk("data/clean/summaries")
else:
    dataset = load_from_disk(os.path.join(project_root, "data/clean/summaries"))
    
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