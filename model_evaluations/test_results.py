import os
import sys
import evaluate

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

def generate_results(pred_file, ref_file):
    preds = getSentences(pred_file)
    refs = getSentences(ref_file)
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
    print("\n")
    
if __name__ == '__main__':
	generate_results(sys.argv[1], sys.argv[2]) #Provide relative path from current directory