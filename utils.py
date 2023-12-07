import os
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def parse_line(line):
    utterance_data, intent_label = line.split(" <=> ")
    items = utterance_data.split()
    words = [item.rsplit(':', 1)[0] for item in items]
    word_labels = [item.rsplit(':', 1)[1] for item in items]
    return {
        'intent_label': intent_label,
        'words': " ".join(words),
        'words_label': " ".join(word_labels),
    }


def create_sentence_tag(data):
    sentences = []
    tags = []
    intent_tags = []
    alltags = []
    for i in range(data.shape[0]):
        sentence = data.iloc[i]['words'].split()
        sentence_tag = data.iloc[i]['words_label'].split()
        intent_tag = data.iloc[i]['intent_label']
        sentences.append(sentence)
        tags.append(sentence_tag)
        intent_tags.append(intent_tag)
        alltags += sentence_tag
    print("<<<--- ATIS Train --->>>")
    print(len(sentences), len(tags), len(intent_tags))
    alltags = list(set(alltags))
    allintents = list(set(intent_tags))
    return sentences, tags, intent_tags, alltags, allintents


def create_tokens_and_labels(id, sample):
    intent = sample['intent']
    utt = sample['utt']
    annot_utt = sample['annot_utt']
    tokens = utt.split()
    labels = []
    label = 'O'
    split_annot_utt = annot_utt.split()
    idx = 0
    BIO_SLOT = False
    while idx < len(split_annot_utt):
        if split_annot_utt[idx].startswith('['):
            label = split_annot_utt[idx].lstrip('[')
            idx += 2
            BIO_SLOT = True
        elif split_annot_utt[idx].endswith(']'):
            if split_annot_utt[idx - 1] == ":":
                labels.append("B-" + label)
                label = 'O'
                idx += 1
            else:
                labels.append("I-" + label)
                label = 'O'
                idx += 1
            BIO_SLOT = False
        else:
            if split_annot_utt[idx - 1] == ":":
                labels.append("B-" + label)
                idx += 1
            elif BIO_SLOT == True:
                labels.append("I-" + label)
                idx += 1
            else:
                labels.append("O")
                idx += 1

    if len(tokens) != len(labels):
        raise ValueError(f"Len of tokens, {tokens}, doesnt match len of labels, {labels}, "
                         f"for id {id} and annot utt: {annot_utt}")
    return tokens, labels, intent


def Read_Massive_dataset(massive_raw):
    sentences_tr, tags_tr, intent_tags_tr = [], [], []
    sentences_val, tags_val, intent_tags_val = [], [], []
    sentences_test, tags_test, intent_tags_test = [], [], []
    all_tags, all_intents = [], []

    for id, sample in enumerate(massive_raw):
        if sample['partition'] == 'train':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_tr.append(tokens)
            tags_tr.append(labels)
            intent_tags_tr.append(intent)
            all_tags += labels

        if sample['partition'] == 'dev':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_val.append(tokens)
            tags_val.append(labels)
            intent_tags_val.append(intent)
            all_tags += labels

        if sample['partition'] == 'test':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_test.append(tokens)
            tags_test.append(labels)
            intent_tags_test.append(intent)
            all_tags += labels

    all_tags = list(set(all_tags))

    allintents = intent_tags_tr + intent_tags_val + intent_tags_test
    all_intents = list(set(allintents))
    return sentences_tr, tags_tr, intent_tags_tr, sentences_val, tags_val, intent_tags_val, sentences_test, tags_test, intent_tags_test, all_tags, all_intents


def parse_data(path):
    lines = path.read_text('utf-8').strip().splitlines()
    df_data = pd.DataFrame([parse_line(line.strip()) for line in lines])
    return df_data


def parse_ourData_newformat(Persian_dir, English_dir, save_dir=None):
    for filename in ["pr_train", "pr_valid", "pr_test", "pr_vocab.intent", "pr_vocab.slot"]:
        path = Path(filename)
        if not path.exists():
            print(f"Downloading {filename}...")
            urlretrieve(Persian_dir + filename + "?raw=true", path)
    for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
        path = Path(filename)
        if not path.exists():
            print(f"Downloading {filename}...")
            urlretrieve(English_dir + filename + "?raw=true", path)


    df_train = parse_data(Path('train'))
    # df_validation = parse_data(Path('valid'))
    # df_test = parse_data(Path('test'))

    pr_df_train = parse_data(Path('pr_train'))
    pr_df_validation = parse_data(Path('pr_valid'))
    pr_df_test = parse_data(Path('pr_test'))

    Data = {}
    sentences_tr, tags_tr, intent_tags_tr, tr_alltags, tr_allintents = create_sentence_tag(df_train)
    pr_sentences_tr, pr_tags_tr, pr_intent_tags_tr, pr_tr_alltags, pr_tr_allintents = create_sentence_tag(pr_df_train)
    sentences_val, tags_val, intent_tags_val, val_alltags, val_allintents = create_sentence_tag(pr_df_validation)
    sentences_test, tags_test, intent_tags_test, test_alltags, test_allintents = create_sentence_tag(pr_df_test)


    Data["tr_inputs"], Data["tr_tags"], Data["tr_intents"] = sentences_tr, tags_tr, intent_tags_tr
    Data["pr_tr_inputs"], Data["pr_tr_tags"], Data["pr_tr_intents"] = pr_sentences_tr, pr_tags_tr, pr_intent_tags_tr
    Data["val_inputs"], Data["val_tags"], Data["val_intents"] = sentences_test, tags_test, intent_tags_test
    Data["test_inputs"], Data["test_tags"], Data["test_intents"] = sentences_val, tags_val, intent_tags_val

    # Data["tr_inputs"] = Data["tr_inputs"] + Data["pr_tr_inputs"]
    # Data["tr_tags"] = Data["tr_tags"] + Data["pr_tr_tags"]
    # Data["tr_intents"] = Data["tr_intents"] + Data["pr_tr_intents"]
    
    Data["tr_tokens"] = Data["tr_inputs"]
    Data["pr_tr_tokens"] = Data["pr_tr_inputs"]
    Data["val_tokens"] = Data["val_inputs"]
    Data["test_tokens"] = Data["test_inputs"]

    alltags = tr_alltags + pr_tr_alltags + val_alltags + test_alltags
    allintents = tr_allintents + pr_tr_allintents + val_allintents + test_allintents
    alltags = list(set(alltags))
    allintents = list(set(allintents))
    dict2 = {}
    dict_rev2 = {}
    inte2 = {}
    inte_rev2 = {}

    for i, tag in enumerate(alltags):
        dict_rev2[tag] = i + 1
        dict2[i + 1] = tag
    print("Slots labels: ", alltags)


    for i, tag in enumerate(allintents):
        inte_rev2[tag] = i + 1
        inte2[i + 1] = tag
    print("Intent labels: ", allintents)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_dir is not None:
        save_obj(Data, save_dir + '/Data')
        save_obj(dict2, save_dir + '/dict2')
        save_obj(dict_rev2, save_dir + '/dict_rev2')
        save_obj(inte2, save_dir + '/inte2')
        save_obj(inte_rev2, save_dir + '/inte_rev2')
        save_obj(alltags, save_dir + '/alltags')


