# Intent Detection and Slot-Filling for Persian Crosslingual Training for Low-resource Languages
This repository contains the code implementation and resources for the paper titled "Intent Detection and Slot-Filling for Persian Crosslingual Training for Low-resource Languages." The paper explores the use of pre-trained language models for intent classification and slot filling tasks in low-resource languages, focusing on Persian. Below is an overview of the contents of this repository:

# Requirements

    transformers >= 4.0
    PyTorch >= 0.4.0, scikit-learn, NumPy
    seqeval for model evaluation
    pytorch-crf

# Usage

## Data Parsing:

    python data_parsing.py

Run this command to parse and read the dataset.
    
## Training:

    python training_script.py

Execute this command to train the model using the defined hyper-parameters.
    
## Evaluation:

    python using_script.py

Use this command to evaluate the trained model.

# Hyper-parameters

You can define hyper-parameters in training_script.py.
