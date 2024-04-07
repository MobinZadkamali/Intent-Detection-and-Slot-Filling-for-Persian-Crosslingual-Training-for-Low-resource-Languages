# Intent Detection and Slot-Filling for Persian Cross-lingual Training for Low-resource Languages
This repository contains the code and resources for the paper titled **"Intent Detection and Slot-Filling for Persian Crosslingual Training for Low-resource Languages."** In this paper, we address the challenges of natural language understanding in low-resource languages by employing pre-trained language models in cross-lingual scenarios.

## Abstract

Intent detection and slot filling are essential tasks for natural language understanding. Deep neural models have shown remarkable performance in these tasks but require large amounts of training data, which is often unavailable in low-resource languages. In this study, we leverage pre-trained language models, specifically Multilingual BERT (mBERT) and XLM-RoBERTa, in various cross-lingual and monolingual scenarios. We translate a portion of the ATIS dataset into Persian to evaluate our proposed models and repeat experiments on the MASSIVE dataset to increase our results’ reliability. Results indicate significant improvements in cross-lingual scenarios over monolingual ones.

## Persian ATIS Dataset

We manually translated and annotated 500 random utterances (approximately 10% of the original data) from the ATIS dataset into Persian. Additionally, we translated the entire test set of the ATIS and included 69 informal translated utterances. The Persian ATIS dataset is divided into train, validation, and test sets. Statistics for these sets in compared with other available datasets are as follows:

| Dataset      | Language       | Vocab Size    | #Train        | #Valid        | #Test         | #Slot         | #Intent
| :---:        |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:      |
| ATIS         | Persian        | 1428          | 500           | 481           | 481           | 130           | 26            |
|              | English        | 5473          | 4478          | 500           | 893           | 130           | 26            |
| MASSIVE      | Persian        | 16432         | 11514         | 2033          | 2974          | 108           | 60            |
|              | English        | 16432         | 11514         | 2033          | 2974          | 108           | 60            |

## Example of a Translated ATIS Utterance with Corresponding Labels

<img src="https://github.com/MobinZadkamali/Intent-Detection-and-Slot-Filling-for-Persian-Crosslingual-Training-for-Low-resource-Languages/assets/37911344/56adafef-9d0e-4b16-8dcb-7c657eacf8bb" width="600" height="400">

## Experimental Results

In the ATIS dataset, the highest value of F1 was obtained in the EN→PR scenario (75.94), the highest accuracy value was achieved in the PR→EN scenario (90.64), and the highest Exact Match value was obtained in the EN+PR mode (50.1). In the MASSIVE dataset, the highest value of all three metrics was attained in the PR→EN scenario (The obtained values for F1, Accuracy, and Exact Match metrics, respectively, are equal to 79.88, 87.79, and 69.87).

Experimental results with all the scenarios, using mBERT pre-trained language model as the encoder on ATIS and MASSIVE test datasets are summarized below:
 
|Strategy | ATIS                                         ||| MASSIVE                                     |||                 
| :---:   |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:      | 
|         |F1              |Accuracy       |Exact Match    | F1            |Accuracy       |Exact Match    |
|EN       |50.96           |86.48          |17.25          |79.68          |87.35          |69.43          |
|PR       |63.54           |76.29          |28.89          |53.15          |32.14          |16.57          |
|PR→EN    |73.59           |**90.64**      |47.19          |**79.88**      |**87.79**      |**69.87**      |
|EN→PR    |74.58           |90.22          |48.44          |79.01          |86.68          |68.72          |
|EN+PR    |**75.59**       |90.22          |**50.1**       |79.61          |87.62          |69.36          |

Experimental results with all the scenarios, using XLM-RoBERTa pre-trained language are presented below:

|Strategy | ATIS                                         ||| MASSIVE                                     |||                 
| :---:   |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:      | 
|         |F1              |Accuracy       |Exact Match    | F1            |Accuracy       |Exact Match    |
|EN       |3.18            |46.77          |0.0            |67.26          |**84.76**      |58.8           |
|PR       |39.78           | 70.27         |8.1            |43.23          |50.2           |25.15          |
|PR→EN    |16.56           |74.63          |1.24           |63.99          |83.01          |55.64          |
|EN→PR    |57.77           |**79.0**       |**23.07**      |65.05          |82.64          |56.82          |
|EN+PR    |**58.56**       |78.17          |21.82          |**68.35**      |83.82          |**59.91**      |

## Comparison with MASSIVE Dataset

To compare the performance of our 10% Persian dataset with larger datasets, we used all 12,664 Persian samples from the MASSIVE dataset. Results for both mBERT and XLM-RoBERTa models are presented below:

|Strategy |       mBERT                                  |||              XLM-RoBERTa                    |||                 
| :---:   |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:      | 
|         |F1              |Accuracy       |Exact Match    | F1            |Accuracy       |Exact Match    |
|PR       |79.31           |87.79          |69.36          |68.91          |85.44          |59.27          |
|PR→EN    |**80.6**        |**87.79**      |**71.47**      |**71.13**      |**86.17**      |**61.97**      |
|EN→PR    |80.22           |87.65          |70.0           |71.06          |85.93          |61.43          |
|EN+PR    |580.12          |87.62          |70.24          |71.06          |85.93          |**61.97**      |

## Conclusion

Our experiments demonstrate the effectiveness of cross-lingual training with pre-trained language models in improving natural language understanding tasks for low-resource languages. "Future research could explore additional techniques to enhance performance in such scenarios.

**Note:** For access to the dataset and detailed experimental results, please refer to the respective files in this repository.

For any inquiries or collaborations, feel free to contact us.

# Usage
## Requirements

    transformers >= 4.0
    PyTorch >= 0.4.0, scikit-learn, NumPy
    seqeval for model evaluation
    pytorch-crf

## Data Parsing:

Run the following command to parse and read the dataset:

    python data_parsing.py
    
## Training:

Execute the following command to train the model using the defined hyperparameters:

    python training_script.py
    
## Evaluation:

Use the following command to evaluate the trained model:

    python using_script.py

## Hyper-parameters

You can define hyper-parameters in **'training_script.py'**.
