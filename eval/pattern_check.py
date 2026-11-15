from eval.run_eval import load_rows, split_rows, DATA_PATH
from app.detectors.regex_layer import PATTERNS, normalize

rows = split_rows(load_rows(DATA_PATH), "dev")
for p in PATTERNS:
    hits = [r["label"] for r in rows if p.search(normalize(r["text"]))]
    print(p.pattern[:50], "| attacks:", sum(hits), "| benign:", len(hits) - sum(hits))