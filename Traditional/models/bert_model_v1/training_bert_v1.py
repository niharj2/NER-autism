from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments
from models.bert_model_v1 import bert_v1
from datasets import load_from_disk


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
data_collator = DataCollatorForTokenClassification(tokenizer)
training_inputs = load_from_disk("Traditional/synthetic_data/tokenised_inputs_asd_updated")
split = training_inputs.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds = split["test"]


trainer = Trainer(
    model=bert_v1.model,
    args=bert_v1.args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)


trainer.train()
trainer.save_model()




