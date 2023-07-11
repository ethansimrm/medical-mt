import os

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

#Load datasets in for training and validation and convert them to an appropriate format
from datasets import load_dataset, Dataset
training_data = load_dataset("ethansimrm/wmt_16_19_22_biomed_train_processed", split = "train") 
validation_data = load_dataset("ethansimrm/wmt_20_21_biomed_validation", split = "validation")
train_data_ready = convertToDictFormat(training_data['text'])
val_data_ready = convertToDictFormat(validation_data['text'])

#Load in our metric
from datasets import load_metric
metric = load_metric("sacrebleu")

#Load correct tokenizer for our model
from transformers import AutoTokenizer
checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

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

#Load in model
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

#Initialise adapter according to Bapna & Firat (2019), with bottleneck dimension 256. This is half of d=512 for this model. 
#Interpretation based on the original paper and in Üstün & Stickland (2022)

from transformers.adapters import AdapterConfig

config = AdapterConfig(
mh_adapter=False, output_adapter=True, #Insert after every FFN layer, but not after every attention layer
reduction_factor=2, #Half of model d=512; largest size we can add while still being a down-projection.  
non_linearity="relu", #Stated in original paper
ln_before = True, #FFN's Layer norm takes place before down-projection
residual_before_ln = True, #Residual connection around entire adapter
original_ln_before = True, #Adapter takes in final output of layer (i.e., after everything is done)
original_ln_after = False, #Neither paper mentions another layer norm after the adapter
init_weights = "bert", #Default weight initialisation
)

model.add_adapter("enfr_adapter", config=config)

import sys
sys.exit()

#Initialise training arguments and loop
from transformers import EarlyStoppingCallback, IntervalStrategy
import torch

batch_size = 32 #Set as high as possible per Popel & Bojar (2018)

training_args = Seq2SeqTrainingArguments( #Collects hyperparameters
    output_dir="opus_wmt_finetuned_enfr_hpc", #Already git cloned and linked to a HF 
    overwrite_output_dir=True, #Overwrite content since directory already exists    
    evaluation_strategy=IntervalStrategy.STEPS, #Evaluates every N steps
    save_steps=4000, #Save model every evaluation
    eval_steps=4000, #Evaluate every ~128k sentences
    logging_steps=4000, #Log compute consumption etc every time we do this	    
    num_train_epochs=16, #About 15mins per eval interval,5 eval intervals per epoch, 24hr compute budget all used in the worst case
    learning_rate=2e-5, #Initial learning rate for AdamW
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01, #Weight decay for loss computation; Loss = Loss + WD * sum (weights squared)
    save_total_limit=1, #Number of checkpoints to save - we just keep the most recent one to resume an interrupted training
    predict_with_generate=True, #Use with ROUGE/BLEU and other translation metrics (see below)
    fp16=True, #Remove fp16 = True if not using CUDA
    #push_to_hub=True, #We will do this manually after fine-tuning is complete
    metric_for_best_model='bleu', #Determines our best model
    load_best_model_at_end=True, #We will retain the best model so far
)

trainer = Seq2SeqTrainer( #Saves us from writing our own training loops
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

trainer.save_model("./opus_wmt_finetuned_enfr_hpc") #Need to save the final model in our local directory
