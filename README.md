# medical-mt
## The Goal
My goal is to create the most accurate neural machine translation model for biomedical translation with minimal resource footprint (time, memory, data, operations, etc.)
## What's Here
For each model I have run, hyperparameters used for model training are stored in that model's associated `.txt` file in `model_configs`, along with comments such as BLEU score, which is computed using [sacreBLEU](https://github.com/mjpost/sacreBLEU), or reasons for infeasibility.
The notebooks in `model_evaluations` record the BLEU score calculations in a reproducible manner.
The scripts I use to fine-tune my models are in `model_runs`.
Finally, the `data` folder stores data from primary sources, such as the [WMT Biomedical Translation Task Repository](https://github.com/biomedical-translation-corpora/corpora).
## Implementation and Reproducibility
All models were trained on [Kaggle's](Kaggle.com) P1000 GPU, and are hosted on [my HuggingFace profile](https://huggingface.co/ethansimrm). All training data has also been uploaded on HuggingFace for easier retrieval. In the future, I will modify my code (and retrain the models) using scripts which can handle multiple GPUs for greater efficiency.
## Current Progress
I am currently performing some (very) preliminary fine-tuning with [T5-Small](https://huggingface.co/t5-small) and [an OPUS-MT English-to-French model](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr). I use the [EMEA English-French Parallel Corpus](https://huggingface.co/datasets/emea) for fine-tuning.
I also use [Google Translate](https://translate.google.co.uk/) as a baseline, as it is used in many clinical settings when human intepreters are unavailable.
I compute the BLEU scores over the abstracts of the WMT16 Biomedical Translation Task EN-FR test set (500 in total) for a quick-and-dirty performance estimate.




