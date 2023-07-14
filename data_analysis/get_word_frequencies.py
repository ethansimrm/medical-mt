import stanza
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
stanza.download('fr', processors='tokenize, mwt', package='sequoia')
nlp_fr = stanza.Pipeline('fr', processors='tokenize, mwt', package='sequoia')
fr_tagged = []
for sentence in train_data_ready['fr']:
  doc = nlp_fr(sentence)
  tokens = [word.text for sent in doc.sentences for word in sent.words] #We just need the text only, not the entire token object.
  fr_tagged.append(tokens)

#Generate a dictionary of tokens and counts 
train_word_frequencies = Counter(fr_tagged)

output = open("train_word_frequencies.txt", "w", encoding = "utf8")
for word in train_word_frequencies.keys():
    output.write(word + "\t" + str(train_word_frequencies[word]) + "\n")
output.close()