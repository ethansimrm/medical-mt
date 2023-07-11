import os
import sys

#User-specified parameters
#GLOSSARY_CHOICES = ["ethansimrm/MeSpEn_enfr_dirty_glossary", "ethansimrm/MeSpEn_enfr_cleaned_glossary"]
#SELECTED_GLOSSARY = GLOSSARY_CHOICES[int(sys.argv[1])]
#DIR_NAMES = ["multiple_glossary", "one_to_one_glossary"]
#OUTPUT_DIR = "opus_wmt_finetuned_enfr_choi_2022_" + DIR_NAMES[int(sys.argv[1])]

MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-tc-big-en-fr"
SELECTED_GLOSSARY = "ethansimrm/MeSpEn_enfr_cleaned_glossary"
OUTPUT_DIR = "opus_big_enfr_FT_choi_2022"

#Explicitly set seed for reproducible behaviour
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

#Import HF token
curr = os.getcwd()
filepath = os.path.join(curr, "../hf_token.txt")
f = open(filepath, "r", encoding = "utf8")
hf_token = f.readline().strip()
f.close()

#Login to HF to extract datasets and push models
from huggingface_hub import login
login(hf_token)

#Assemble and shuffle training data in situ
from datasets import load_dataset, Dataset
glossary = load_dataset(SELECTED_GLOSSARY, split = "train")
unchanged_train = load_dataset("ethansimrm/choi_unchanged_train", split = "train")
annotated_train = load_dataset("ethansimrm/choi_annotated_sampled_train", split = "train")

glossary_ready = convertToDictFormat(glossary['text'])
unchanged_train_ready = convertToDictFormat(unchanged_train['text'])
annotated_train_ready = convertToDictFormat(annotated_train['text'])

from datasets import concatenate_datasets
full_train_unshuffled = concatenate_datasets([glossary_ready, unchanged_train_ready, annotated_train_ready])
train_data_ready = full_train_unshuffled.shuffle(seed=42).flatten_indices()

#Load validation data in and convert to an appropriate format as well
validation_data = load_dataset("ethansimrm/wmt_20_21_biomed_validation", split = "validation")
val_data_ready = convertToDictFormat(validation_data['text'])

#Load in our metric
from datasets import load_metric
metric = load_metric("sacrebleu")

#Load correct tokenizer for our model, along with the model itself
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
checkpoint = MODEL_CHECKPOINT
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#Ensure the tokeniser and model can recognise our special tokens and handle them appropriately
special_tokens_dict = {'additional_special_tokens': ['<term_start>','<term_end>','<term_trans>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

#Tokenise data
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

#Builds batches from dataset
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

#Define the metric we wish to compute over our validation set, along with some simply space-stripping
import numpy as np

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

#Initialise training arguments and loop
from transformers import EarlyStoppingCallback, IntervalStrategy
import torch

batch_size = 16 #Set as high as possible per Popel & Bojar (2018)

training_args = Seq2SeqTrainingArguments( 
    output_dir=OUTPUT_DIR, 
    overwrite_output_dir=True,    
    evaluation_strategy=IntervalStrategy.STEPS, 
    save_steps=8000, 
    eval_steps=8000, 
    logging_steps=8000, 	    
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

trainer = Seq2SeqTrainer( 
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

trainer.save_model(OUTPUT_DIR)
