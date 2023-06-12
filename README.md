# medical-mt
## The Goal
My goal is to create the most accurate neural machine translation model for biomedical translation with minimal resource footprint (time, memory, data, operations, etc.). 
## What's Here
Hyperparameters used for model training will be stored in a `.txt` file in `model_configs`, along with comments such as BLEU score, which is computed using [sacreBLEU](https://github.com/mjpost/sacreBLEU), or reasons for infeasibility.
The scripts in `model_evaluations` will record the BLEU score calculations in a reproducible manner.
The scripts I use to fine-tune my models will be in `model_runs`.
The `data` directory stores data from primary sources, such as the [WMT Biomedical Translation Task Repository](https://github.com/biomedical-translation-corpora/corpora). 
In every directory mentioned so far, the `early_experiments` directory archives my very rough initial experiments.
The `data_processing` directory will host the scripts I use to process primary data into .txt files adhering to a common format. This format is source TAB target NEWLINE. Any output files are also saved, depending on whether they are intermediate or fully processed - this is just in case something breaks.
## Implementation and Reproducibility
All models will be trained on [Kaggle's](Kaggle.com) P1000 GPU, and are hosted on [my HuggingFace profile](https://huggingface.co/ethansimrm). All raw data files, and all scripts/notebooks used, will be hosted here. Note that in some scripts, some features, such as [Weights and Biases](https://wandb.ai/) for model training statistics, may be specific to Kaggle. 
All generated .txt files will also be uploaded on my HuggingFace profile. I will process these into a convenient format for training (i.e., a `Dataset` object) subsequently.
## Current Progress
I will be finetuning a pre-trained English-to-French model from [OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) with data provided by the WMT Biomedical Translation Task organisers. Right now, I've only just gathered the raw data files, and my next step will be pre-processing and alignment. More updates soon!





