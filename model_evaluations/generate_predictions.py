import sys
from transformers import pipeline, AutoTokenizer, set_seed
from tqdm import tqdm

#For reproducibility
set_seed(42)

def read_in_queries(input_file):
    f = open(input_file, 'r', encoding = 'utf-8')
    queries = [line.strip() for line in f.readlines()]
    f.close()
    return queries

def generate_results(HF_model, input_file, output_file):
    test = read_in_queries(input_file)
    try:
        model = pipeline("translation_en_to_fr", model = HF_model)
    except:
        model = pipeline("translation_en_to_fr", model = HF_model, tokenizer = AutoTokenizer.from_pretrained(HF_model, src_lang="en", tgt_lang="fr"))
    finally:
        preds = [model(s) for s in tqdm(test)]
        f = open(output_file, 'w', encoding = 'utf-8')
        for line in preds:
            f.write(line[0]['translation_text']+ '\n')
        f.close()

if __name__ == '__main__':
	generate_results(sys.argv[1], sys.argv[2], sys.argv[3]) #Provide HF model name and relative paths from current directory