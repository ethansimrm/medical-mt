import spacy
from datasets import load_dataset, Dataset
from collections import Counter

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

#Tokenise FR portion of training set at the word level
fr_tagger = spacy.load("fr_dep_news_trf")
fr_tagged = []
for sentence in train_data_ready['fr']:
    fr_tagged_sent = fr_tagger(sentence)
    fr_tagged_tokenised = [token.text for token in fr_tagged_sent]
    fr_tagged += fr_tagged_tokenised

#Generate a dictionary of tokens and counts 
train_word_frequencies = Counter(fr_tagged)

output = open("train_word_frequencies.txt", "w", encoding = "utf8")
for word in train_word_frequencies.keys():
    output.write(word + "\t" + str(train_word_frequencies[word]) + "\n")
output.close()