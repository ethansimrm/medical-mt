from datasets import load_dataset, Dataset

#Converts data in src [TAB] tgt [NEWLINE] format to a format suitable for model training
def convertToDictFormat(data):
    source = []
    target = []
    for example in data:
        example = example.strip()
        sentences = example.split("\t")
        source.append(sentences[0])
        target.append(sentences[1])
    ready = Dataset.from_dict({"en":source, "fr":target})
    return ready
    
train_data = load_dataset("ethansimrm/wmt_16_19_22_biomed_train_processed", split = "train")
train_data_ready = convertToDictFormat(train_data['text'])

from transformers import AutoTokenizer
opus_base_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
opus_big_checkpoint = "Helsinki-NLP/opus-mt-tc-big-en-fr"
opus_base_tokenizer = AutoTokenizer.from_pretrained(opus_base_checkpoint)
opus_big_tokenizer = AutoTokenizer.from_pretrained(opus_big_checkpoint)

#Tokenise training data using both tokenizers
source_lang = 'en'
target_lang = 'fr'

def tokenize_base(examples):
    inputs = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = opus_base_tokenizer(inputs, text_target=targets, padding="longest")
    #Pad to longest sequence in batch, no truncation - we filtered too-long sentences out already
    return model_inputs

def tokenize_big(examples):
    inputs = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = opus_big_tokenizer(inputs, text_target=targets, padding="longest")
    #Pad to longest sequence in batch, no truncation - we filtered too-long sentences out already
    return model_inputs

tokenized_train_base = train_data_ready.map(tokenize_base, batched=True)
tokenized_train_big = train_data_ready.map(tokenize_big, batched=True)

BASE_SPECIAL_TOKENS = [0, 1, 59513]
BIG_SPECIAL_TOKENS = [43311, 50387, 43312, 53016]

#We will obtain per-token counts for both tokenised datasets.
from collections import Counter
def get_token_counts(tokenized_dataset):
    all_tokens = []
    for token_group in tokenized_dataset:
        all_tokens += token_group
    return Counter(all_tokens)

train_base_token_counts = get_token_counts(tokenized_train_base['labels'])
train_big_token_counts = get_token_counts(tokenized_train_big['labels'])

#Get rid of special tokens
for unwanted_token_id in BASE_SPECIAL_TOKENS:
    del train_base_token_counts[unwanted_token_id]

for unwanted_token_id in BIG_SPECIAL_TOKENS:
    del train_big_token_counts[unwanted_token_id]

def compute_token_freq(train_token_counts):
    freq_in_train = {}
    total_train_tokens = sum(train_token_counts.values())
    for tok_id in train_token_counts.keys():
        freq_in_train[tok_id] = train_token_counts[tok_id] / total_train_tokens
    return freq_in_train

base_token_train_freq = compute_token_freq(train_base_token_counts)
big_token_train_freq = compute_token_freq(train_big_token_counts)

def write_dict_to_file(filename, dict_to_write):
  output = open(filename, "w", encoding = "utf8")
  for key in dict_to_write.keys():
    output.write(str(key) + "\t" + str(dict_to_write[key]) + "\n")
  output.close()
  return

write_dict_to_file("train_token_base_freq.txt", base_token_train_freq)
write_dict_to_file("train_token_big_freq.txt", big_token_train_freq) 