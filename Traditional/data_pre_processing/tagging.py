import pandas as pd
import numpy as np 

tags = {
    "O": 0,
    "B-ACTY": 1,  "I-ACTY": 2,
    "B-DORA": 3,  "I-DORA": 4,
    "B-DSYN": 5,  "I-DSYN": 6,
    "B-FNDG": 7,  "I-FNDG": 8,
    "B-HLCA": 9,  "I-HLCA": 10,
    "B-INBE": 11, "I-INBE": 12,
    "B-MENP": 13, "I-MENP": 14,
    "B-MOBD": 15, "I-MOBD": 16,
    "B-PODG": 17, "I-PODG": 18,
    "B-QLCO": 19, "I-QLCO": 20,
    "B-SOCB": 21, "I-SOCB": 22,
    "B-SOSY": 23, "I-SOSY": 24
}

flipped_tags = {}
for key, value in tags.items():
  flipped_tags[value] = key

flipped_tags



df = pd.read_json("Traditional/synthetic_data/asd_clinical_ner_12types_10000.jsonl", lines = True)
df["words"] = np.nan
df["ner_tags"] = np.nan



all_words = []
all_ner_tags = []

for words in df["bio"]:
  word_list = []
  ner_tags = []

  for word, tag in words:
    word_list.append(word)
    ner_tags.append(tags[tag])

  all_words.append(word_list)
  all_ner_tags.append(ner_tags)


df["words"] = all_words
df["ner_tags"] = all_ner_tags


df.to_json('Traditional/synthetic_data/asd_clinical_ner_12types_10000_processed.jsonl', orient="records", lines=True)

