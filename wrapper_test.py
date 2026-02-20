from datasets import load_dataset

print("Test 1: Biomni-Eval1")
d1 = load_dataset("biomni/Eval1", split="test")
print(f"OK: {len(d1)} tasks")

print("Test 2: LAB-Bench")
d2 = load_dataset("futurehouse/lab-bench", "LitQA2", split="train")
print(f"OK: {len(d2)} tasks")

print("Done!")
