from datasets import Dataset, load_dataset
import pandas as pd
import langid, string
from tqdm import tqdm

def generate_dataset(HF_repo): #Assumed to be public
    data = load_dataset(HF_repo, split = "train") #Split is "train" if unspecified in repo name; we've already split validation and test sentences
    #Assumed that data is in SRC [TAB] TGT [NEWLINE] format
    source = []
    target = []
    for example in training_data['text']:
        example = example.strip()
        sentences = example.split("\t")
        source.append(sentences[0])
        target.append(sentences[1])
    ready_data = Dataset.from_dict({"en":source, "fr":target})
    return ready_data

def retain_sentence(row): #This defines a good sentence we wish to retain
    word_counts = []
    #Exclude repeats and empty source or target sentences
    if ((row["en"] == row["fr"]) or (row["en"] == "") or (row["fr"] == "")):
        return False
    for sentence in (row["en"], row["fr"]):
        #Exclude sentences with mismatched () or '' or ""
        if ((sentence.count("(") != sentence.count(")")) or 
            (sentence.count("'") % 2 != 0) or
            (sentence.count('"') % 2 != 0)):
            return False
        sent_length = len(sentence)
        #Exclude sentences with punctuation percentage > 0.4
        if (count(sentence,set(string.punctuation)) > 0.4 * sent_length):
            return False
        #Exclude sentences with > 150 words
        num_words = len(sentence.split(" "))
        if (num_words > 150):
            return False
        word_counts.append(num_words)
        #Exclude sentences with char-to-word ratio > 12 or < 1.5
        c2w_ratio = sent_length / num_words
        if ((c2w_ratio > 12) or (c2w_ratio < 1.5)):
            return False
    #Heuristic "alignment" filtering
    word_ratio = word_counts[0] / word_counts[1]
    if((word_ratio >= 9) or (1/word_ratio >= 9)):
        return False
    #Expensive language-determining step
    if ((langid.classify(row['en'])[0] != "en") or 
    (langid.classify(row['fr'])[0] != "fr")):
        return False
    return True

def generate_preprocessed_data(HF_repo, output_file):
    #Housekeeping
    langid.set_languages(["en", "fr"]) #Constrain language set
    tqdm.pandas() #Monitor progress
    #Preprocessing pipeline
    ready_data = generate_dataset(HF_repo)
    tempDataset = pd.DataFrame(ready_data)
    tempDataset_dedup = tempDataset.drop_duplicates() #Get rid of duplicate sentences
    tempDataset_goodSentences = tempDataset_dedup.progress_apply(retain_sentence, axis=1) #Apply per-sentence filtering
    filtered_set = tempDataset_dedup[tempDataset_goodSentences].reset_index(drop=True) #Avoid retaining additional column indices
    filtered_set = filtered_set.sample(frac=1).reset_index(drop=True) #Shuffle to avoid overfitting 
    f = open(output_file, "w", encoding = "utf-8")
    source = filtered_set["en"]
    target = filtered_set["fr"]
    for i in range(len(filtered_set)):
        f.write(source[i] + "\t" + target[i] + "\n")
    f.close()
    
if __name__ == '__main__':
	generate_preprocessed_data(sys.argv[1], sys.argv[2]) #Provide HF repository name and output .txt file name