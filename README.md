# medical-mt
## The Goal
My goal is to create the most accurate neural machine translation model for biomedical translation with minimal resource footprint (time, memory, data, operations, etc.). 
## What's Here
Hyperparameters used for model training are stored in a `.txt` file in `model_configs`.
The scripts in `model_evaluations` generate predictions from models and compute metrics over these predictions reproducibly.
The scripts I use to fine-tune my models are in `model_runs`.
The `data` directory stores data from primary sources, such as the [WMT Biomedical Translation Task Repository](https://github.com/biomedical-translation-corpora/corpora). 
In every directory mentioned so far, the `early_experiments` directory archives my very rough initial experiments.
The `data_processing` directory hosts the scripts I used to align primary data, along with my data preprocessing pipeline.
## Implementation and Reproducibility
All models have been trained on [Kaggle's](Kaggle.com) P1000 GPU, and are hosted on [my HuggingFace profile](https://huggingface.co/ethansimrm). All raw data files, and all scripts/notebooks used, will be hosted here. Note that in some scripts, some features, such as [Weights and Biases](https://wandb.ai/) for model training statistics, may be specific to Kaggle. 
All data files are also uploaded on my HuggingFace profile. I process these into a convenient format for training (i.e., a `Dataset` object) in situ, due to issues with HuggingFace's programmatic upload.
I use two virtual environments; the one described by `scispacey_requirements.txt` uses [Conda](https://www.anaconda.com/download) and runs Python 3.8 specifically for [SciSpacy](https://allenai.github.io/scispacy/). The other one runs Python 3.10.11 and uses a normal venv.   
## Current Progress
I am currently finetuning a pre-trained English-to-French model from [OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) with data provided by the WMT Biomedical Translation Task organisers. I've aligned and preprocessed my training data, and computed results from five metrics (BLEU, CHRF++, TER, METEOR, and three flavours of Terminology Usage Rate) over this model's predictions. 
## TODO
- Porting my code over to a more powerful GPU cluster, while still retaining compute consumption.
- Implementing a better way to translate biomedical-domain information.
 





