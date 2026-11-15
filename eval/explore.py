import random

try:
    from .run_eval import DATA_PATH, load_rows, split_rows
except ImportError:  # pragma: no cover - allows running the file directly
    from run_eval import DATA_PATH, load_rows, split_rows

rows = split_rows(load_rows(DATA_PATH), "dev")
attacks = [r["text"] for r in rows if r["label"] == 1]
benign = [r["text"] for r in rows if r["label"] == 0]

random.seed(1)
print("--- ATTACKS ---")
for t in random.sample(attacks, 25):
    print("-", t[:200].replace("\n", " "))
print("\n--- BENIGN ---")
for t in random.sample(benign, 15):
    print("-", t[:200].replace("\n", " "))