import os,sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import numpy as np
from models.transformer.seqTagger import transformertagger
from models.transformer.BERT_Joint_IDSF import BertIDSF

import project_statics
from utils import load_obj, save_obj
import itertools
from seqeval.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch
import time

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
    t_l = true_labels
    p_l = predicted_labels
    for c,[k,i,j] in enumerate(zip(toks, t_l, p_l)):
      if len(i) != len(j):
        true_labels.pop(c)
        predicted_labels.pop(c)
    print("Test Slots F1: ",f1_score(true_labels, predicted_labels))
    print("Test Intents Accuracy: ",accuracy_score(true_intents, predicted_intents))
    EM = 0
    for i in range(len(predicted_labels)):
        if accuracy_score(true_labels[i], predicted_labels[i])==1 and true_intents[i]==predicted_intents[i]:
            EM+=1
    print('Test Sentence Accuracy: ',EM/len(predicted_labels))

    # now we use it
    start_time1 = time.time()
    # toks, labels, intents = tagger_obj.get_label(["what days of the week do flights from san jose to nashville fly on", "please list flights from st. louis to st. paul which depart after 10 am thursday morning"], need_tokenization=True)
    toks, labels, intents = tagger_obj.get_label(["پروازهای صبح شنبه از تورنتو به واشنگتن را لیست کن ","در فرودگاه شارلوت چند هواپیمای مختلف از هواپیمایی یواس وجود دارد"], need_tokenization=True)
    end_time1 = time.time()
    print("Required time to calculate intent and slot labels for 2 samples: ",end_time1-start_time1)

print("Golden Labels\n")
# print("what:O are:O the:O flights:O from:O tacoma:B-fromloc.city_name to:O san:B-toloc.city_name jose:I-toloc.city_name <=> atis_flight")
# print("please:O list:O flights:O from:O st.:B-fromloc.city_name louis:I-fromloc.city_name to:O st.:B-toloc.city_name paul:I-toloc.city_name which:O depart:O after:B-depart_time.time_relative 10:B-depart_time.time am:I-depart_time.time thursday:B-depart_date.day_name morning:B-depart_time.period_of_day <=> atis_flight")
print("پروازهای:O صبح:B-depart_time.period_of_day شنبه:B-depart_date.day_name از:O تورنتو:B-fromloc.city_name به:O واشنگتن:B-toloc.city_name را لیست:O کن:O <=> flight")
print("در:O فرودگاه:B-airport_name شارلوت:I-airport_name چند:O هواپیمای:O مختلف:O از:O هواپیمایی:B-airline_name یواس:B-airline_name وجود:O دارد:B-airline_name <=> aircraft")
for i in range(len(toks)):
  print("\n## Intent:", intents[i])
  print("## Slots:")
  for token, slot in zip(toks[i],  labels[i]):
      print(f"{token:>10} : {slot}")



