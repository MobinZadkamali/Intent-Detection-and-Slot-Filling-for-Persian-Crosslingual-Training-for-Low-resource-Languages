# Intent-Detection-and-Slot-Filling-for-Persian-Crosslingual-Training-for-Low-resource-Languages
PyTorch implementation for experiments in the paper: Intent Detection and Slot Filling for Persian Crosslingual Training for Low-resource Languages

# Requirements

    transformers >= 4.0
    PyTorch >= 0.4.0, scikit-learn, NumPy
    seqeval to evaluate model
    pytorch-crf

# Usage

    Data Parsing:
    ```python
    python data_parsing.py
    ```
    Run this command to parse and read the dataset.
    
    Training:
    ```python
    python training_script.py
    ```
    Execute this command to train the model using the defined hyper-parameters.
    
    Evaluation:
    ```python
    python using_script.py
    ```
    Use this command to evaluate the trained model.

# Hyperparameters

    You can define hyperparameters in training_script.py.
