# Intent Detection and Slot-Filling for Persian Crosslingual Training for Low-resource Languages
This repository contains the code implementation and resources for the paper titled "Intent Detection and Slot-Filling for Persian Crosslingual Training for Low-resource Languages." The paper explores the use of pre-trained language models for intent classification and slot filling tasks in low-resource languages, focusing on Persian. Below is an overview of the contents of this repository:
Abstract

The abstract of the paper outlines the motivation, methodology, and key findings of the research.
Dataset

The Persian ATIS dataset used in the experiments is available for download. It includes train, validation, and test sets, along with statistics presented in Table Y.
Figures

Figure X illustrates examples of translated ATIS utterances with corresponding labels.
Tables

Experimental results for different scenarios, using both mBERT and XLM-RoBERTa pre-trained language models, are provided in tables Z and N. Additionally, Table M presents the results of comparing the performance of the Persian dataset with comparable data in two languages.
Code
Requirements

    transformers >= 4.0
    PyTorch >= 0.4.0, scikit-learn, NumPy
    seqeval for model evaluation
    pytorch-crf

Usage

    Data Parsing: Execute python data_parsing.py to parse and read the dataset.
    Training: Run python training_script.py to train the model using the defined hyper-parameters.
    Evaluation: Use python using_script.py for model evaluation.
    
PyTorch implementation for experiments in the paper: Intent Detection and Slot Filling for Persian Crosslingual Training for Low-resource Languages

# Requirements

    transformers >= 4.0
    PyTorch >= 0.4.0, scikit-learn, NumPy
    seqeval to evaluate model
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
