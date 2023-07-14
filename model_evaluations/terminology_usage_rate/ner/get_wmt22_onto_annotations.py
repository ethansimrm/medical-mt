#Many thanks to the helpful example at https://github.com/ncbo/ncbo_rest_sample_code/blob/master/python/python3/annotate_text.py

import urllib.request, urllib.error, urllib.parse
import json
import os
import pandas as pd
from tqdm import tqdm

REST_URL = "http://data.bioontology.org"
API_KEY = "73b4f33e-4540-4e6e-b05c-7b1a699c4c2c"
PREFERENCE_STRING = "&longest_only=false&exclude_numbers=false&whole_word_only=true&exclude_synonyms=false"

def get_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())

dir_path = os.getcwd()
f = open(os.path.join(dir_path, "../../wmt22test.txt"), "r", encoding = "utf8")
en_sent = [line.strip() for line in f.readlines()]
f.close()

#Get annotations
annotations_en = []
for sentence in tqdm(en_sent):
    # Annotate using the provided text and query all ontologies
    annotations = get_json(REST_URL + "/annotator?text=" + urllib.parse.quote(sentence) + PREFERENCE_STRING)
    annotations_per_sentence = set() #Ignore duplicates
    for result in annotations:
        annotations_per_sentence.add(result["annotations"][0]["text"])
    annotations_en.append(list(annotations_per_sentence))

#Generate results
sent_IDs = []
annotations = []
for i in range(len(annotations_en)):
    for annotation in annotations_en[i]:
        sent_IDs.append(i)
        annotations.append(annotation)
term_list = pd.DataFrame(data = {"sent_ID" : sent_IDs, "annotation" : annotations})
term_list.to_csv("wmt22_onto_annotations.txt", sep = "\t", header = False, index = False)