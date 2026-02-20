from transformers import AutoTokenizer
from data_set_for_hugging_face import load_data
from data_set_for_hugging_face import tokenize_and_align


train_ds, test_ds = load_data()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


training_inputs = train_ds.map(lambda batch: tokenize_and_align(batch, tokenizer), batched = True)
training_inputs.save_to_disk("Traditional/synthetic_data/tokenised_inputs_asd_updated")

