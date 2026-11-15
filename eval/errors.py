import random
from collections import Counter
from eval.run_eval import DATA_PATH , split_rows , load_rows
from app.detectors.regex_layer import regex_detect as detect

rows = split_rows(load_rows(DATA_PATH) , "dev")
missed = [r["text"] for r in rows if r["label"] == 1 and not detect(r["text"])]
false_alarm = [r["text"] for r in rows if r["label"] == 0 and detect(r["text"])]

print(f"----- MISSED -----{len(missed)}--")
for t in random.Random(1).sample(missed, 40):
    print("-" , t[:160].replace("\n"," "))

print(f"------FALSE ALARM-----{len(false_alarm)}")
for t in false_alarm:
    print("-" , t[:160].replace("\n"," "))

STOP = {"the", "a", "in", "you", "your", "an", "ai", "of","i", "and", "as" , "by" , "are", "this" , "that" ,"how", "when" , "be" , "what" , "to" , "my", "is", "for", "with" , "if", "do" ,"it", "just"}
print("-----MOST COMMON WORDS-----")
words = Counter(w for t in missed for w in set(t.lower().split()) if w not in STOP)
print(words.most_common(40))


