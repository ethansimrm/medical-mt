# medical-mt
## The Goal
My goal is to find a better way to translate biomedical-domain information, with a specific focus on terminology translation.
## What's Here
The `data` directory stores data from primary sources, such as the [WMT Biomedical Translation Task Repository](https://github.com/biomedical-translation-corpora/corpora). 
The `data_analysis` directory hosts the scripts I used to analyse my training sets to find out more about them.
The `data_processing` directory hosts the scripts I used to align primary data, along with my data preprocessing pipeline.
The scripts in `model_evaluations` generate predictions from models and compute metrics over these predictions reproducibly. These metrics include Terminology Usage Rate, and the workflow I used to extract terminology from the test set is hosted here. Model predictions and results are hosted here too.
The actual scripts I use to fine-tune my models, and their logs, are hosted in `model_runs`.
The `deprecated` and `early_experiments` folders - wherever you see them - host the various experiments and trials I subsequently abandoned.
There are two sets of virtual environment requirements. `venv_requirements.txt` has everything I required throughout this project - including all abandoned experiments and trials - while `rcs_hpc_venv_requirements.txt` only contains those dependencies strictly necessary for model fine-tuning.
## Implementation and Reproducibility
All models are trained on a single Nvidia RTX6000 GPU provided by the [Imperial College London Research Computing Service](https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/), and are hosted on [my HuggingFace profile](https://huggingface.co/ethansimrm). All raw data files, and all scripts/notebooks used, will be hosted here. 
All data files are also uploaded on my HuggingFace profile. I process these into a convenient format for training (i.e., a `Dataset` object) in situ, due to issues with HuggingFace's programmatic upload.
## Script-specific Features
In some early scripts, some features, such as [Weights and Biases](https://wandb.ai/) for model training statistics, are specific to Kaggle, and may not be available on other environments. These scripts are not crucial to result generation and can be ignored.
Additionally, some scripts involve Huggingface Token input, because I thought this was necessary to download my hosted datasets. Much later, I found that this was not the case, and that section can be safely removed if you do not wish to push your own models to Huggingface using these scripts.
## Current Progress
I am currently finetuning [two pre-trained](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) [English-to-French models](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fr) from OPUS-MT using data provided by the WMT Biomedical Translation Task organisers. I also use an in-domain glossary provided by [MeSpEn](https://github.com/PlanTL-GOB-ES/MeSpEn_Glossaries). 






