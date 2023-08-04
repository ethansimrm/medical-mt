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
    
#We will load in our two tokenisers, the training data, and the three glossaries under consideration - the whole glossary, 0.1 sampled, and 0.25 sampled.
train_data = load_dataset("ethansimrm/wmt_16_19_22_biomed_train_processed", split = "train")
gloss_entire = load_dataset("ethansimrm/MeSpEn_enfr_cleaned_glossary", split = "train")
gloss_025 = load_dataset("ethansimrm/sampled_glossary_0.25_train", split = "train")
gloss_010 = load_dataset("ethansimrm/sampled_glossary_0.1_train", split = "train")
train_data_ready = convertToDictFormat(train_data['text'])
gloss_entire_ready = convertToDictFormat(gloss_entire['text'])
gloss_025_ready = convertToDictFormat(gloss_025['text'])
gloss_010_ready = convertToDictFormat(gloss_010['text'])

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

#We will also tokenise the 0.1 and 0.25 glossaries, because these were used by the WCE methods on the base model.
gloss_010_fr = gloss_010_ready['fr']
gloss_010_base_tokens = opus_base_tokenizer(text_target=gloss_010_fr)["input_ids"]

gloss_025_fr = gloss_025_ready['fr']
gloss_025_base_tokens = opus_base_tokenizer(text_target=gloss_025_fr)["input_ids"]

#Take note of special tokens inserted by the tokenizer, and which are therefore not organically present in the training set.
BASE_SPECIAL_TOKENS = [0, 1, 59513]
BIG_SPECIAL_TOKENS = [43311, 50387, 43312, 53016]

#We will obtain per-token counts for all three tokenised datasets.
from collections import Counter
def get_token_counts(tokenized_dataset):
    all_tokens = []
    for token_group in tokenized_dataset:
        all_tokens += token_group
    return Counter(all_tokens)
    
train_base_token_counts = get_token_counts(tokenized_train_base['labels'])
gloss_010_base_token_counts = get_token_counts(gloss_010_base_tokens)
gloss_025_base_token_counts = get_token_counts(gloss_025_base_tokens)

#Get rid of special tokens
for unwanted_token_id in BASE_SPECIAL_TOKENS:
    del train_base_token_counts[unwanted_token_id]
    del gloss_010_base_token_counts[unwanted_token_id]
    del gloss_025_base_token_counts[unwanted_token_id]
    
#Now, we will obtain token frequencies.
def compute_token_freq(glossary_token_counts, train_token_counts):
    freq_in_train = {}
    total_train_tokens = sum(train_token_counts.values())
    for tok_id in glossary_token_counts.keys():
        try:
            freq_in_train[tok_id] = train_token_counts[tok_id] / total_train_tokens
        except:
            freq_in_train[tok_id] = 0 #In case of KeyError being raised
    return freq_in_train

gloss_010_base_token_train_freq = compute_token_freq(gloss_010_base_token_counts, train_base_token_counts)
gloss_025_base_token_train_freq = compute_token_freq(gloss_025_base_token_counts, train_base_token_counts)
    
def write_dict_to_file(filename, dict_to_write):
  output = open(filename, "w", encoding = "utf8")
  for key in dict_to_write.keys():
    output.write(str(key) + "\t" + str(dict_to_write[key]) + "\n")
  output.close()
  return

write_dict_to_file("gloss_010_base.txt", gloss_010_base_token_train_freq)
write_dict_to_file("gloss_025_base.txt", gloss_025_base_token_train_freq)

#Repeat this for the opus_big tokenizer, and our other glossary
gloss_010_big_tokens = opus_big_tokenizer(text_target=gloss_010_fr)["input_ids"]

gloss_entire_fr = gloss_entire_ready['fr']
gloss_entire_big_tokens = opus_big_tokenizer(text_target=gloss_entire_fr)["input_ids"]

train_big_token_counts = get_token_counts(tokenized_train_big['labels'])
gloss_010_big_token_counts = get_token_counts(gloss_010_big_tokens)
gloss_entire_big_token_counts = get_token_counts(gloss_entire_big_tokens)

for unwanted_token_id in BIG_SPECIAL_TOKENS:
    del train_big_token_counts[unwanted_token_id]
    del gloss_010_big_token_counts[unwanted_token_id]
    del gloss_entire_big_token_counts[unwanted_token_id]

gloss_010_big_token_train_freq = compute_token_freq(gloss_010_big_token_counts, train_big_token_counts)
gloss_entire_big_token_train_freq = compute_token_freq(gloss_entire_big_token_counts, train_big_token_counts)

write_dict_to_file("gloss_010_big.txt", gloss_010_big_token_train_freq)
write_dict_to_file("gloss_entire_big.txt", gloss_entire_big_token_train_freq)