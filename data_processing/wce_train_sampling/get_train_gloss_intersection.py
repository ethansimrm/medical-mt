import sys

#Parameters chosen by user
TRAIN_PROPORTION_LIST = [0.25, 0.5, 0.75, 1.0]
TRAIN_SAMPLE = TRAIN_PROPORTION_LIST[int(sys.argv[1])]
OUTPUT_FILE = "glossary_intersect_train_sampled_" + str(TRAIN_SAMPLE) + ".txt"

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
    
#Sample training data    
from datasets import load_dataset, Dataset
training_data = load_dataset("ethansimrm/wmt_16_19_22_biomed_train_processed", split = "train") 
train_data_ready = convertToDictFormat(training_data['text'])
if (TRAIN_SAMPLE != 1.0):
  train_sampled = train_data_ready.train_test_split(train_size = TRAIN_SAMPLE, seed = 42)["train"] #Seed for reproducibility
else:
  train_sampled = train_data_ready

#Load in glossary
term_candidates = load_dataset("ethansimrm/MeSpEn_enfr_cleaned_glossary", split = "train")
terms_ready = convertToDictFormat(term_candidates['text'])

#Find intersection - we need both en and fr terms. This is the slow part.
intersection = set()
for bitext in train_sampled:
  for term in terms_ready:
    if ((term['en'], term['fr']) in intersection): #Lookup and skip
      continue
    if ((term['en'] in bitext['en']) and (term['fr'] in bitext['fr'])):
      intersection.add((term['en'], term['fr']))
      
#Write output to file
output = open(OUTPUT_FILE, "w", encoding = "utf8")
for bitext in intersection:
  output.write(bitext[0] + "\t" + bitext[1] + "\n")
output.close()