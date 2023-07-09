# medical-mt
## The Goal
My goal is to find a better way to translate biomedical-domain information.
## What's Here
The scripts in `model_evaluations` generate predictions from models and compute metrics over these predictions reproducibly.
The scripts I use to fine-tune my models are in `model_runs`.
The `data` directory stores data from primary sources, such as the [WMT Biomedical Translation Task Repository](https://github.com/biomedical-translation-corpora/corpora). 
The `data_processing` directory hosts the scripts I used to align primary data, along with my data preprocessing pipeline.
## Implementation and Reproducibility
All models are trained on a single Nvidia RTX6000 GPU, and are hosted on [my HuggingFace profile](https://huggingface.co/ethansimrm). All raw data files, and all scripts/notebooks used, will be hosted here. Note that in some scripts, some features, such as [Weights and Biases](https://wandb.ai/) for model training statistics, are specific to Kaggle, and may not be available on other environments.
All data files are also uploaded on my HuggingFace profile. I process these into a convenient format for training (i.e., a `Dataset` object) in situ, due to issues with HuggingFace's programmatic upload.
I use two virtual environments; the one described by `scispacy_requirements.txt` uses [Conda](https://www.anaconda.com/download) and runs Python 3.8 specifically for [SciSpacy](https://allenai.github.io/scispacy/). The other one runs Python 3.10.11 and uses a normal venv.   
## Current Progress
I am currently finetuning [two pre-trained](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) [English-to-French models](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fr) from OPUS-MT, and an [M2M100](https://huggingface.co/facebook/m2m100_418M) model from Facebook, using data provided by the WMT Biomedical Translation Task organisers. I also use an in-domain glossary provided by [MeSpEn](https://github.com/PlanTL-GOB-ES/MeSpEn_Glossaries). 






