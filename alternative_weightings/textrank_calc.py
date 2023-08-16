import spacy
import pytextrank
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

#Sample 10% of corpus sentences due to space/efficiency issues - building a graph is very expensive. We will take a random sample a la Ailem for this.
train_data = load_dataset("ethansimrm/wmt_16_19_22_biomed_train_processed", split = "train")
train_data_ready = convertToDictFormat(train_data['text'])
train_sampled = train_data_ready.train_test_split(train_size = 0.1, seed = 42)["train"]

#Create one big "document" from our corpus to obtain corpus-level textrank
corpus = ""
for sent in train_sampled['fr']:
    corpus += sent + " "

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop #We will feed this into the stop words recogniser and the scrubber function
fr_stop_words = list(fr_stop)

#Generate stop words dictionary
sw = spacy.load("fr_dep_news_trf")
stop_words = {}
for word in fr_stop_words:
    doc = sw(word)
    for tok in doc:
        stop_words[tok.text] = [tok.pos_]

#Scrubber function to remove determiners and pronouns (la, le ... )
from spacy.tokens import Span
@spacy.registry.misc("prefix_scrubber")
def prefix_scrubber():
    def scrubber_func(span: Span) -> str:
        for token in span:
            if token.pos_ not in ["DET", "PRON"]:
                break
            span = span[1:]
        return span.text
    return scrubber_func

nlp = spacy.load("fr_dep_news_trf") #We will then run through the sampled corpus using textrank.
nlp.add_pipe("textrank", config={"stopwords" : stop_words, "scrubber": {"@misc": "prefix_scrubber"}})
nlp.max_length = 7000000 #7m chars for 10%

doc = nlp(corpus, disable = ["ner"]) #Kick out NER to make it run faster

output = open("textrank_phrases_and_scores.txt", "w", encoding = "utf-8")
for phrase in doc._.phrases:
    output.write(str(phrase.text) + "\t" + str(phrase.rank) + "\n")
output.close()