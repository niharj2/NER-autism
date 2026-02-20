from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from data_pre_processing import tagging


model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path="bert-base-cased", num_labels = len(tagging.tags), id2label = tagging.flipped_tags, label2id = tagging.tags)


args = TrainingArguments(
    output_dir="bert-finetuned-ner",
    eval_strategy="epoch",      # or evaluation_strategy="epoch" depending on your version
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)


