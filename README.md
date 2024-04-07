# Intent Detection and Slot-Filling for Persian Cross-lingual Training for Low-resource Languages
This repository contains the code and resources for the paper titled "Intent Detection and Slot-Filling for Persian Crosslingual Training for Low-resource Languages." In this paper, we address the challenges of natural language understanding in low-resource languages by employing pre-trained language models in cross-lingual scenarios.

## Abstract

Intent detection and slot filling are essential tasks for natural language understanding. Deep neural models have shown remarkable performance in these tasks but require large amounts of training data, which is often unavailable in low-resource languages. In this paper, we leverage pre-trained language models, specifically Multilingual BERT (mBERT) and XLM-RoBERTa, in various cross-lingual and monolingual scenarios. We translate a portion of the ATIS dataset into Persian to evaluate our proposed models and repeat experiments on the MASSIVE dataset for robustness. Results indicate significant improvements in cross-lingual scenarios over monolingual ones.

## Dataset

We manually translated and annotated 500 random utterances (approximately 10% of the original data) from the ATIS dataset into Persian. Additionally, we translated the entire test set of the ATIS and included 69 informal translated utterances. The Persian ATIS dataset is divided into train, validation, and test sets. Statistics for these sets are as follows:

| Dataset      | Language       | Vocab Size    | #Train        | #Valid        | #Test         | #Slot         | #Intent
| :---:        |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:      |
| ATIS         | Persian        | 1428          | 500           | 481           | 481           | 130           | 26            |
|              | English        | 5473          | 4478          | 500           | 893           | 130           | 26            |
| MASSIVE      | Persian        | 16432         | 11514         | 2033          | 2974          | 108           | 60            |
|              | English        | 16432         | 11514         | 2033          | 2974          | 108           | 60            |

## Example Translations

![ATIS_new](https://github.com/MobinZadkamali/Intent-Detection-and-Slot-Filling-for-Persian-Crosslingual-Training-for-Low-resource-Languages/assets/37911344/56adafef-9d0e-4b16-8dcb-7c657eacf8bb)

## Experimental Results

Experimental results using mBERT and XLM-RoBERTa pre-trained language models are presented below.

Results using mBERT as the encoder on ATIS and MASSIVE test datasets are summarized in Table Z.
XLM-RoBERTa Results

Results using XLM-RoBERTa as the encoder on ATIS and MASSIVE test datasets are summarized in Table N.
Insights

    mBERT yields superior performance across both datasets.
    The EN!PR scenario in the ATIS dataset achieves the highest F1 score (75.94), while the PR!EN scenario achieves the highest accuracy (90.64).
    In the MASSIVE dataset, the PR!EN scenario outperforms others, with F1, accuracy, and exact match metrics of 79.88, 87.79, and 69.87, respectively.

# Comparison with MASSIVE Dataset

To compare the performance of our 10% Persian dataset with larger datasets, we used all 12,664 Persian samples from the MASSIVE dataset. Results for both mBERT and XLM-RoBERTa models are presented below:



## Conclusion

Our experiments demonstrate the effectiveness of cross-lingual training with pre-trained language models in improving natural language understanding tasks for low-resource languages. Further research can explore additional techniques to enhance performance in such scenarios.

**Note:** For access to the dataset and detailed experimental results, please refer to the respective files in this repository.

For any inquiries or collaborations, feel free to contact us.

# Usage
## Requirements

    transformers >= 4.0
    PyTorch >= 0.4.0, scikit-learn, NumPy
    seqeval for model evaluation
    pytorch-crf

## Data Parsing:

    python data_parsing.py

Run this command to parse and read the dataset.
    
## Training:

    python training_script.py

Execute this command to train the model using the defined hyper-parameters.
    
## Evaluation:

    python using_script.py

Use this command to evaluate the trained model.

## Hyper-parameters

You can define hyper-parameters in training_script.py.
