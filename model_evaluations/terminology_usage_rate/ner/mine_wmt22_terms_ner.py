import stanza
from simalign import SentenceAligner
import string
import pandas as pd

#Open our test files first
f = open("wmt22test.txt", "r", encoding = "utf8")
en_sent = [line.strip() for line in f.readlines()]
f.close()
f = open("wmt22gold.txt", "r", encoding = "utf8")
fr_sent = [line.strip() for line in f.readlines()]
f.close()

#Tokenise and pos-tag FR sentences first using sequoia-trained treebank model. We don't need anything more.
stanza.download('fr', processors='tokenize, mwt, pos', package='sequoia')
nlp_fr = stanza.Pipeline('fr', processors='tokenize, mwt, pos', package='sequoia')

#Extract only the information we need - PoS tags and text.
fr_tagged = []
for sentence in fr_sent:
    doc = nlp_fr(sentence)
    tokens = [{"text" : word.text, "upos" : word.upos} for sent in doc.sentences for word in sent.words]
    fr_tagged.append(tokens)

#Now, initialise our EN NER pipeline. We will take the set union of all NER models Stanza has to offer.
stanza.download('en', processors= {"ner": ["i2b2", "Radiology", "AnatEM", "BC5CDR", "BC4CHEMD", "BioNLP13CG", "JNLPBA", "Linnaeus", "ncbi_disease", "S800"]} , package='mimic')
nlp_en = stanza.Pipeline('en', package = 'mimic', processors= {"ner": ["i2b2", "Radiology", "AnatEM", "BC5CDR", "BC4CHEMD", "BioNLP13CG", "JNLPBA", "Linnaeus", "ncbi_disease", "S800"]})

en_tagged = []
for sentence in en_sent:
    doc = nlp_en(sentence)
    tokens = [token for sent in doc.sentences for token in sent.tokens] #Extract NER-annotated tokens
    words = [{"text":word.text, "upos":word.upos, "ner":token.ner} for token in tokens for word in token.words] #Account for multi-word tokens
    en_tagged.append(words)

#Word alignment
aligner = SentenceAligner(matching_methods="a") #Argmax only; the simAlign paper stated that this gives the best performance for English-French alignment
alignments_list = []
for i in range(len(en_tagged)):
    src_sent = [token["text"]for token in en_tagged[i]]
    tgt_sent = [token["text"] for token in fr_tagged[i]] 
    alignments = aligner.get_word_aligns(src_sent, tgt_sent)
    alignments_list.append(alignments["inter"])

#Query alignments using token indices
prospective_ne_fr = []
for i in range(len(alignments_list)):
    ne_fr_per_sentence = []
    for alignment in alignments_list[i]:
        src_word = en_tagged[i][alignment[0]]
        tgt_word = fr_tagged[i][alignment[1]]
        if ((src_word["ner"] != 'O') & #Word has an NER tag
            (src_word["upos"] in ["PROPN", "NOUN", "ADJ", "VERB"]) & #Determiners, adverbs, prepositions, etc. are not terminologies of interest, because we are doing word-level checking
            (tgt_word["upos"] == src_word["upos"]) & #Ensure both pipelines agree on PoS tags
            (src_word["text"] not in string.punctuation) &
            (tgt_word["text"] not in string.punctuation)): #In case of incorrect tagging wrt punctuation
                ne_fr_per_sentence.append(tgt_word["text"])
    prospective_ne_fr.append(ne_fr_per_sentence)

#At this point we almost have a candidate terminology. Before we do anything further, let's dataframe this and spit it out for review.
sent_IDs = []
terms = []
for i in range(len(prospective_ne_fr)):
    for term in prospective_ne_fr[i]:
        sent_IDs.append(i)
        terms.append(term)
term_list = pd.DataFrame(data = {"sent_ID" : sent_IDs, "term" : terms})
term_list.to_csv("wmt22_ner_terms.txt", sep = "\t", header = False, index = False)


