

def tokenize_and_align(batch, tokenizer):
  
  tokenize = tokenizer(batch["words"], is_split_into_words=True, truncation=True)
  labels = []

  for i, word_label in enumerate(batch["ner_tags"]):
    word_ids = tokenize.word_ids(i) # this would give us the tags of words in each sentence
    new = []
    prev = None

    for word_id in word_ids:
      if word_id is None:
        new.append(-100)

      elif word_id != prev:
        new.append(word_label[word_id])

      else:
        lab = word_label[word_id]

        if lab % 2 == 1: # convert from B to I
          lab += 1

        new.append(lab)

      prev = word_id

    labels.append(new)

    
  tokenize["labels"] = labels
  return tokenize





