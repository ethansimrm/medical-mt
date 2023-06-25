import os
import sys
import evaluate
import pandas as pd
import spacy
from tqdm import tqdm

dir_path = os.getcwd()

#The four metrics we will evaluate
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")
meteor = evaluate.load('meteor')

#Read .txt into list
def getSentences(filename):
    f = open(os.path.join(dir_path, filename), "r", encoding = "utf8")
    sentences = [line.strip() for line in f.readlines()]
    f.close()
    return sentences

#Generate global list of predicted tokens
def generate_tokens(pred_list):
    fr_tagger = spacy.load("fr_dep_news_trf")
    global fr_tagged 
    fr_tagged = []
    for sentence in tqdm(pred_list):
        fr_tagged_sent = fr_tagger(sentence)
        fr_tagged_tokenised = [token.text for token in fr_tagged_sent] #We just need the text only, not the entire token object. 
        fr_tagged.append(fr_tagged_tokenised)

#Rule-based checker which optionally ignores casing errors and partitive article errors, depending on stringency.
def find_term_in_sentence(row, stringency):
    if (stringency == "strict"):
        query = row["fr_term"]
        total_occurrences = len([found for found in fr_tagged[row["sent_ID"]] if query == found])
    elif (stringency == "caseless"):
        query = row["fr_term"].lower()
        total_occurrences = len([found for found in fr_tagged[row["sent_ID"]] if query.lower() == found.lower()])
    elif (stringency == "loose"):
        query = row["fr_term"].lower()
        total_occurrences = len([found for found in fr_tagged[row["sent_ID"]] if (query.lower() == found.lower()) or 
                                 ("d" + query.lower() == found.lower()) or 
                                 ("l" + query.lower() == found.lower())])
    else:
        print("Invalid stringency!")
        return
    if (total_occurrences <= row["counts"]): #Account for predicting some but not all
        return total_occurrences
    else: #Over-prediction should be ignored - other metrics will suffer if this is severe.
        return row["counts"]

#Terminology usage metric expressed as proportion of found terminologies for all three stringency levels.
def generate_TUR(pred_list, term_refs):
    
    #Check for terms
    generate_tokens(pred_list)
    term_refs["matches_strict"] = term_refs.apply(find_term_in_sentence, args=("strict",), axis=1)
    term_refs["matches_caseless"] = term_refs.apply(find_term_in_sentence, args=("caseless",), axis=1)
    term_refs["matches_loose"] = term_refs.apply(find_term_in_sentence, args=("loose",), axis=1)
    term_refs.to_csv("found_terminology.txt", sep="\t", index=False, header=True)
    
    #Compute statistics
    total = term_refs["counts"].sum()
    exact_matches = term_refs["matches_strict"].sum()
    print("Total Terminologies: ", total)
    print("Exact Matches: ", exact_matches)
    print({"tur": exact_matches/total })
    
    non_cased_matches = term_refs["matches_caseless"].sum()
    print("Non-cased Matches: ", non_cased_matches)
    print({"tur_caseless": non_cased_matches/total })
    
    non_cased_matches_loose = term_refs["matches_loose"].sum()
    print("Non-cased Matches ignoring partitive article errors: ", non_cased_matches_loose)
    print({"tur_loose": non_cased_matches_loose/total})
        
def generate_results(pred_file, ref_file, terms_file):
    preds = getSentences(pred_file)
    refs = getSentences(ref_file)
    term_refs = pd.read_csv(terms_file, sep = "\t", header = None, names = ["sent_ID", "fr_term", "counts"]) #Default encoding is UTF-8
    print("\n\nResults are:\n\n")
    result = bleu.compute(predictions = preds, references = refs) #BLEU score for provided input and references
    result = {"bleu": result["score"]}
    print(result)
    result = chrf.compute(predictions = preds, references = refs, word_order = 2) #Include word bigrams for CHRF++
    result = {"CHRF++":result["score"]}
    print(result)
    result = ter.compute(predictions = preds, references = refs, case_sensitive = True) #Casing is important - treat as an edit error
    result = {"ter": result["score"]}
    print(result)
    result = meteor.compute(predictions = preds, references = refs)
    print(result)
    generate_TUR(preds, term_refs)
    print("\n")
    
if __name__ == '__main__':
	generate_results(sys.argv[1], sys.argv[2], sys.argv[3]) #Provide relative path from current directory