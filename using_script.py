import os,sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import numpy as np
from models.transformer.seqTagger import transformertagger
from models.transformer import seqTagger
from models.transformer.BERT_Joint_IDSF import BertIDSF
import project_statics
from utils import load_obj, save_obj
import itertools
from seqeval.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
import time
import itertools
import seaborn as sn
import pandas as pd
import pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # now we use it
    save_path = 'test_IDSF_bert'
    data_path = project_statics.SFID_pickle_files
    Data = load_obj(data_path + '/Data')
    dict2 = load_obj(data_path + '/dict2')
    inte2 = load_obj(data_path + '/inte2')

    tagger_obj = transformertagger(save_path, BertIDSF, dict2, inte2, device=torch.device("cpu"))

    # calculating test results
    test_texts = []
    start_time = time.time()
    toks, predicted_labels, predicted_intents = tagger_obj.get_label(Data["test_inputs"], need_tokenization=False)
    end_time = time.time()
    print("Required time to calculate intent and slot labels for 700 samples: ",end_time-start_time)

    true_labels = Data["test_tags"]
    true_intents = Data["test_intents"]

    seqTagger.test_printer('/content/drive/MyDrive/Joint_IDSF/AISE_SNIPS.txt', toks, true_labels, predicted_labels, true_intents, predicted_intents)


    print("Test Slots F1: ",f1_score(true_labels, predicted_labels))
    print("Test Intents Accuracy: ",accuracy_score(true_intents, predicted_intents))
    EM = 0
    for i in range(len(predicted_labels)):
        if accuracy_score(true_labels[i], predicted_labels[i])==1 and true_intents[i]==predicted_intents[i]:
            EM+=1
    print('Test Sentence Accuracy: ',EM/len(predicted_labels))
   
    cm = confusion_matrix(list(itertools.chain(*true_labels)), list(itertools.chain(*predicted_labels)), labels=np.unique(list(itertools.chain(*true_labels))))
    df_cm = pd.DataFrame(cm, index = np.unique(list(itertools.chain(*true_labels))),
                  columns = np.unique(list(itertools.chain(*true_labels))))
    open_file = open('/content/drive/MyDrive/Joint_IDSF/df_cm.pkl', "wb")
    pickle.dump(df_cm, open_file)
    open_file.close()
    fig, ax = plt.subplots(figsize=(30,30))
    sn.heatmap(df_cm, annot=True, ax=ax, fmt='g')
    fig.savefig('/content/drive/MyDrive/Joint_IDSF/cm1.png')   # save the figure to file
    plt.close(fig) 

    cm = confusion_matrix(true_intents, predicted_intents, labels=np.unique(true_intents))
    df_cm = pd.DataFrame(cm, index = np.unique(true_intents),
                  columns = np.unique(true_intents))
    sn.set(font_scale=1.7) 
    fig, ax = plt.subplots(figsize=(30,30))
    sn.heatmap(df_cm, annot=True, ax=ax, fmt='g')
    fig.savefig("/content/drive/MyDrive/Joint_IDSF/cm2.png")   # save the figure to file
    plt.close(fig) 

    # now we use it
#     start_time1 = time.time()
#     # toks, labels, intents = tagger_obj.get_label(["what days of the week do flights from san jose to nashville fly on", "please list flights from st. louis to st. paul which depart after 10 am thursday morning"], need_tokenization=True)
#     end_time1 = time.time()
#     print("Required time to calculate intent and slot labels for 2 samples: ",end_time1-start_time1)

# print("Golden Labels\n")
# # print("what:O are:O the:O flights:O from:O tacoma:B-fromloc.city_name to:O san:B-toloc.city_name jose:I-toloc.city_name <=> atis_flight")
# # print("please:O list:O flights:O from:O st.:B-fromloc.city_name louis:I-fromloc.city_name to:O st.:B-toloc.city_name paul:I-toloc.city_name which:O depart:O after:B-depart_time.time_relative 10:B-depart_time.time am:I-depart_time.time thursday:B-depart_date.day_name morning:B-depart_time.period_of_day <=> atis_flight")

# for i in range(len(toks)):
#   print("\n## Intent:", intents[i])
#   print("## Slots:")
#   for token, slot in zip(toks[i],  labels[i]):
#       print(f"{token:>10} : {slot}")



