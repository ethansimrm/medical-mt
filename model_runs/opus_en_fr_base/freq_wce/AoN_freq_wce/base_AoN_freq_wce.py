import os
import sys
import numpy as np

#Parameters chosen by user

GLOSSARY_LIST = ["ethansimrm/MeSpEn_enfr_cleaned_glossary", "ethansimrm/sampled_glossary_1.0_train"]
CHOSEN_GLOSSARY = GLOSSARY_LIST[int(sys.argv[1])]
GLOSS_NAME_LIST = ["unsampled", "train-sampled"]
GLOSS_NAME = GLOSS_NAME_LIST[int(sys.argv[1])]
PROPORTION_LIST = [0.2, 0.4, 0.6, 0.8]
PROPORTION = PROPORTION_LIST[int(sys.argv[2])]
WEIGHT_LIST = [1.25, 1.5, 1.75, 2.0]
WEIGHT = WEIGHT_LIST[int(sys.argv[3])]

OUTPUT_DIR = "opus_base_adapt_wce_gloss_" + GLOSS_NAME + "_prop_" + str(PROPORTION) + "_weight_" + str(WEIGHT)

MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-fr"
SPECIAL_TOKEN_IDS = [0, 1, 59513] #</s>, <unk>, <pad>, which we do not want to up-weight

#Explicitly set seed to be 42 for reproducible behaviour
from transformers import set_seed
set_seed(42)

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

#Load datasets in for training and validation and convert them to an appropriate format
from datasets import load_dataset, Dataset
training_data = load_dataset("ethansimrm/wmt_16_19_22_biomed_train_processed", split = "train") 
validation_data = load_dataset("ethansimrm/wmt_20_21_biomed_validation", split = "validation")
train_data_ready = convertToDictFormat(training_data['text'])
val_data_ready = convertToDictFormat(validation_data['text'])

#Also load in our glossary
term_candidates = load_dataset(CHOSEN_GLOSSARY, split = "train")
terms_ready = convertToDictFormat(term_candidates['text'])

#Load in our metric
from datasets import load_metric
metric = load_metric("sacrebleu", experiment_id = OUTPUT_DIR) #Avoid concurrency issues

#Load correct tokenizer for our model
from transformers import AutoTokenizer
checkpoint = MODEL_CHECKPOINT
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#Tokenise training and validation data (we'd have done this later anyway)
source_lang = 'en'
target_lang = 'fr'

def preprocess_function(examples):
    inputs = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, text_target=targets, padding="longest")
    #Pad to longest sequence in batch, no truncation - we filtered too-long sentences out already
    return model_inputs

tokenized_train = train_data_ready.map(preprocess_function, batched=True)
tokenized_val = val_data_ready.map(preprocess_function, batched=True)

#Obtain counts for each unique FR token across the training set
from collections import Counter
aggregated_train_tokens = []
for token_group in tokenized_train["labels"]:
    aggregated_train_tokens += token_group
unique_train_tokens = Counter(aggregated_train_tokens)

#Process glossary using tokeniser and obtain FR token counts in the same manner
terms_ready_fr = terms_ready['fr']
glossary_tokens = tokenizer(text_target=terms_ready_fr)["input_ids"]

from collections import Counter
aggregated_glossary_tokens = []
for token_group in glossary_tokens:
    aggregated_glossary_tokens += token_group
unique_glossary_tokens = Counter(aggregated_glossary_tokens)

#Remove special tokens - we want tokens organic to our data
for unwanted_token_id in SPECIAL_TOKEN_IDS:
  del unique_train_tokens[unwanted_token_id]
  del unique_glossary_tokens[unwanted_token_id]
  
#Log number of occurrences of each glossary token in the training set
keys_to_remove = []
glossary_tokens_freq = unique_glossary_tokens
for key in glossary_tokens_freq.keys():
    if (unique_train_tokens[key] == 0): #Tokens not found in the training set will have value 0 - remove them
      keys_to_remove.append(key)
    else:
      glossary_tokens_freq[key] = unique_train_tokens[key]

for absent_key in keys_to_remove:
  del glossary_tokens_freq[absent_key]

#Sort keys in ascending order of counts and select a favoured subset of the rarest tokens to upweight to WEIGHT
glossary_tokens_freq = {k: v for k, v in sorted(glossary_tokens_freq.items(), key=lambda item: item[1])}
sorted_token_ids = list(glossary_tokens_freq.keys())
num_chosen_tokens = round(PROPORTION * len(sorted_token_ids))
chosen_token_ids = sorted_token_ids[:num_chosen_tokens]

#Generate weight vector
import torch

#All vocabulary labels (tokens) are weighted equally initially
weights = np.ones(len(tokenizer), dtype=np.float32) 

for tok_id in chosen_token_ids:
    weights[tok_id] = WEIGHT

#We also need to override the compute_loss function in the Seq2SeqTrainer to weight our negative log-likelihood loss.
from transformers import Seq2SeqTrainer
from torch import nn

class WCETrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        #Get labels (i.e., correct answers), convert to double, host on same device
        labels = inputs.get("labels").type(torch.LongTensor).to(model.device) 
        # forward pass
        outputs = model(**inputs) #Loss is pre-computed; we need the raw logits
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=model.device))
        #Reshape logits to (N, C) and labels to (N) to conform to CEL input, where N = n(tokens) in batch
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

#Builds batches from dataset
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

#Define the metric we wish to compute over our validation set, along with some simply space-stripping
def postprocess_text(preds, labels): 
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True) #Convert back into words

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) #Ignore padded labels added by the data collator to the test set
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) 

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels) #Remove leading and trailing spaces

    result = metric.compute(predictions=decoded_preds, references=decoded_labels) #BLEU score for provided input and references
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens) #Compute mean prediction length
    result = {k: round(v, 4) for k, v in result.items()} #Round score to 4dp
    return result

#Load in model
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

#Initialise training arguments and loop
from transformers import EarlyStoppingCallback, IntervalStrategy

batch_size = 32 #Set as high as possible per Popel & Bojar (2018)

training_args = Seq2SeqTrainingArguments( 
    output_dir=OUTPUT_DIR,  
    overwrite_output_dir=True,  
    evaluation_strategy=IntervalStrategy.STEPS,
    save_steps=4000, 
    eval_steps=4000, 
    logging_steps=4000, 	    
    num_train_epochs=16,
    learning_rate=2e-5, 
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01, 
    save_total_limit=1, 
    predict_with_generate=True, 
    fp16=True, 
    metric_for_best_model='bleu', 
    load_best_model_at_end=True, 
)

trainer = WCETrainer( 
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.1)]
    #If we don't improve by at least 0.1 BLEU on the validation set for 5 evaluations, we stop training
)

trainer.train()

#trainer.save_model(OUTPUT_DIR) #Save space