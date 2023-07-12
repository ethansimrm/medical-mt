import os

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
filepath = os.path.join(curr, "../../hf_token.txt")
f = open(filepath, "r", encoding = "utf8")
hf_token = f.readline().strip()
f.close()

#Login to HF to extract datasets and push models
from huggingface_hub import login
login(hf_token)

#Load sampled dataset and convert to an appropriate format
from datasets import load_dataset, Dataset
sampled_data = load_dataset("ethansimrm/choi_sampled_train", split = "train") 
sampled_data_ready = convertToDictFormat(sampled_data['text'])

#Load in our PoS taggers - we choose transformers for accuracy in tokenisation and PoS-tagging
import spacy
en_tagger = spacy.load("en_core_web_trf")
fr_tagger = spacy.load("fr_dep_news_trf")

#Tokenise and PoS-tag sentences
en_tagged = []
for sentence in sampled_data_ready['en']:
    en_tagged_sent = en_tagger(sentence)
    en_tagged_tokenised = [token for token in en_tagged_sent] #Access PoS tag information later on
    en_tagged.append(en_tagged_tokenised)
    
fr_tagged = []
for sentence in sampled_data_ready['fr']:
    fr_tagged_sent = fr_tagger(sentence)
    fr_tagged_tokenised = [token for token in fr_tagged_sent]
    fr_tagged.append(fr_tagged_tokenised)
    
#Load in word aligner
from simalign import SentenceAligner
aligner = SentenceAligner(matching_methods="a") 
#Argmax only; the simAlign paper stated that this gives the best performance for English-French alignment

#Word-align our sentences
alignments_list = []
for i in range(len(en_tagged)):
    src_sent = [token.text for token in en_tagged[i]]
    tgt_sent = [token.text for token in fr_tagged[i]] 
    alignments = aligner.get_word_aligns(src_sent, tgt_sent)
    alignments_list.append(alignments["inter"])
    
#Crawl through our alignments looking for noun-noun pairings
output = open("noun-alignments.txt", "w", encoding = "utf8")
for i in range(len(alignments_list)):
    for aligned_pair in alignments_list[i]:
        src_ind = aligned_pair[0]
        tgt_ind = aligned_pair[1]
        src_tok = en_tagged[i][src_ind]
        tgt_tok = fr_tagged[i][tgt_ind]
        if ((src_tok.pos_ == "NOUN") and (tgt_tok.pos_ == "NOUN")):
          output.write(str(i) + "\t" + src_tok.text + "\t" + tgt_tok.text + "\n")
          #Include sentence numbers for max-three accounting later
output.close()
          