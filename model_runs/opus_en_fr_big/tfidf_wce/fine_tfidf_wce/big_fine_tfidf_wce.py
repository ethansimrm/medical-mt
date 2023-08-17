import os
import sys
import numpy as np

#Parameters chosen by user
CHOSEN_GLOSSARY = "ethansimrm/MeSpEn_enfr_cleaned_glossary"
PRECISION = 2
UPPER_BOUND_WEIGHT = 1.5
OUTPUT_DIR = "opus_big_fine_tfidf_wce_unsampled"

#GLOSSARY_LIST = ["ethansimrm/MeSpEn_enfr_cleaned_glossary", "ethansimrm/sampled_glossary_1.0_train"]
#CHOSEN_GLOSSARY = GLOSSARY_LIST[int(sys.argv[1])]
#GLOSS_NAME_LIST = ["unsampled", "train-sampled"]
#GLOSS_NAME = GLOSS_NAME_LIST[int(sys.argv[1])]
#PRECISION_LIST = [0, 1, 2, 3]
#PRECISION = PRECISION_LIST[int(sys.argv[2])]
#UPPER_BOUND_WEIGHT_LIST = [1.25, 1.5, 1.75, 2.0]
#UPPER_BOUND_WEIGHT = UPPER_BOUND_WEIGHT_LIST[int(sys.argv[3])]
#OUTPUT_DIR = "opus_big_adapt_wce_gloss_" + GLOSS_NAME + "_precision_" + str(PRECISION) + "_ubweight_" + str(UPPER_BOUND_WEIGHT)

MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-tc-big-en-fr"
SPECIAL_TOKEN_IDS = [43311, 50387, 43312, 53016] #Respectively </s>, <unk>, <s>, and <pad>, which we do not want to up-weight

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
    
#Converts TFIDF data in src [TAB] tgt [NEWLINE] format to a format suitable for model training
def extractTFIDFWords(data):
    words = []
    scores = []
    for example in data:
        example = example.strip()
        sentences = example.split("\t")
        words.append(sentences[0]) 
        scores.append(float(sentences[1])) #Keep scores here while retaining ordinal information
    return (words, scores)

#Load datasets in for training, validation and TFIDF data, and convert them to an appropriate format
from datasets import load_dataset, Dataset
training_data = load_dataset("ethansimrm/wmt_16_19_22_biomed_train_processed", split = "train") 
validation_data = load_dataset("ethansimrm/wmt_20_21_biomed_validation", split = "validation")
tfidf_words = load_dataset("ethansimrm/train_tfidf_words_and_weights", split = "train")

train_data_ready = convertToDictFormat(training_data['text'])
val_data_ready = convertToDictFormat(validation_data['text'])
tfidf_words_ready, tfidf_scores = extractTFIDFWords(tfidf_words['text'])

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

#Process glossary using tokeniser and obtain unique FR tokens
terms_ready_fr = terms_ready['fr']
glossary_tokens = tokenizer(text_target=terms_ready_fr)["input_ids"]

from collections import Counter
aggregated_glossary_tokens = []
for token_group in glossary_tokens:
    aggregated_glossary_tokens += token_group
unique_glossary_tokens = Counter(aggregated_glossary_tokens)

#Remove special tokens - we want tokens organic to our data
for unwanted_token_id in SPECIAL_TOKEN_IDS:
  del unique_glossary_tokens[unwanted_token_id]
  
gloss_token_list = list(unique_glossary_tokens.keys())

#Construct an ordered list of tokens and scores based on TFIDF data
def getOrderedTokenAndScoreList(words, scores, model_tokeniser, special_tokens):
    found_tok = set()
    ordered_tok = []
    tok_scores = []
    for idx, word in enumerate(words):
        tokenised_word = model_tokeniser(text_target=word)["input_ids"]
        word_score = scores[idx] #Assign the most important parent word's score to each token
        for tok in tokenised_word:
            if tok in special_tokens:
                continue
            elif tok not in found_tok:
                ordered_tok.append(tok)
                tok_scores.append(word_score)
                found_tok.add(tok)
    return (ordered_tok, tok_scores)
    
tfidf_ordered_tokens, tfidf_ordered_scores = getOrderedTokenAndScoreList(tfidf_words_ready, tfidf_scores, tokenizer, SPECIAL_TOKEN_IDS)

#Only take those tokens found in our glossary - by design, no matter the glossary, we will only take tokens which exist in the training set
sorted_token_ids = []
sorted_token_scores = []
for idx, token in enumerate(tfidf_ordered_tokens):
  if token in gloss_token_list: #Preserve order
    sorted_token_ids.append(token) 
    sorted_token_scores.append(round(np.log(tfidf_ordered_scores[idx]), PRECISION)) #Apply the rounded log calculation - we are confident that no scores are 0
    
#Obtain unique RL(TFIDF) values in ASCENDING order
unique_rltfidf_values = list(Counter(sorted_token_scores).keys())
unique_rltfidf_values.sort()

#Generate all possible weights
num_unique_rltfidf = len(unique_rltfidf_values)
band_weights = np.linspace(1.0, UPPER_BOUND_WEIGHT, num = num_unique_rltfidf).tolist() #Generates 1.0 ... UPPER_BOUND_WEIGHT at regular intervals in ascending order; there are num_unique_rltfidf elements here

#Associate sorted RL(TFIDF) values with respective weights - the higher the RL(TFIDF) value (i.e., the TFIDF weight), the higher the weight
#This means that the first value - the lowest - should get the lowest weight
#This is distinct from RLF, where we wish to assign the highest weight to the term with the lowest frequency (therefore, RLF)
rltfidf_weight_assoc = {}
for idx, rltfidf in enumerate(unique_rltfidf_values):
  rltfidf_weight_assoc[rltfidf] = band_weights[idx]
  
#Associate token ids with weights based on RLF value as above  
weight_assignments = {}
for idx, tok in enumerate(sorted_token_ids):
  weight_assignments[tok] = rltfidf_weight_assoc[sorted_token_scores[idx]] #This is a specific element of band_weights
  
#At this point, we have a dictionary of per-token weight assignments, but we want to be extra careful and remove all the "1.0" values in case of minute discrepancies
wa_copy = weight_assignments
lowest_weight_value = min(weight_assignments.values())
weight_assignments = {k:v for k,v in wa_copy.items() if v != lowest_weight_value}

#Generate weight vector
import torch

#All vocabulary labels (tokens) are weighted equally initially
weights = np.ones(len(tokenizer), dtype=np.float32) 

for tok_id in weight_assignments.keys():
    weights[tok_id] = weight_assignments[tok_id]

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

trainer.save_model(OUTPUT_DIR)
