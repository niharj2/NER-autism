from datasets import Dataset
import pandas as pd 

def load_data():
    df = pd.read_json("Traditional/synthetic_data/asd_clinical_ner_12types_10000_processed.jsonl", lines = True)
    ds = Dataset.from_pandas(df[["id", "words", "ner_tags"]])
    return ds["train"], ds["test"]
