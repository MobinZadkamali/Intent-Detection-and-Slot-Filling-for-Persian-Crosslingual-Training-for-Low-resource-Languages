from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import itertools
import torch.nn.functional as F


def get_ouputs(model, dataloader: DataLoader, device: torch.device):
    """

    computes for evaluation phase in training

    :param PreTrainedModel model: Bert\RoBERTa for token classification
    :param dataloader: dataloader containing the data
    :param device: torch device
    :return:
    """
    final_layer_is_crf = if_final_layer_is_crf(model.__class__)
    # Put the model into evaluation mode
    model.eval()

    predictions_slots, true_labels_slots, predictions_intents, true_labels_intents = [], [], [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_intents, lens = batch

        sorted_indexes = torch.flipud(torch.argsort(lens))
        lens = torch.index_select(lens, 0, sorted_indexes)
        b_input_ids = torch.index_select(b_input_ids, 0, sorted_indexes)[:, :max(lens).to(dtype=torch.long, device=device)]
        b_input_mask = torch.index_select(b_input_mask, 0, sorted_indexes)[:, :max(lens).to(dtype=torch.long, device=device)]
        b_labels = torch.index_select(b_labels, 0, sorted_indexes)[:, :max(lens).to(dtype=torch.long, device=device)]
        b_intents = torch.index_select(b_intents, 0, sorted_indexes)

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, lens=lens, device=device)
        # Move logits and labels to CPU
        intent_logits = outputs[1][0]
        slot_logits = outputs[1][1]
  
        if type(intent_logits) == list:
            intent_logits = np.array(intent_logits)
        else:
            intent_logits = intent_logits.detach().cpu().numpy()

        # Calculate the accuracy for this batch of test sentences.
        # eval_loss += outputs[0].mean().item()

        if not final_layer_is_crf:
            if type(slot_logits) == list:
                slot_logits = np.array(slot_logits)
            else:
                slot_logits = slot_logits.detach().cpu().numpy()
            predictions_slots.extend([list(p) for p in np.argmax(slot_logits, axis=2)])
        else:
            # CRf does not need argmax
            predictions_slots.extend(np.array(model.crf.decode(slot_logits)))

        label_ids = b_labels.to('cpu').numpy()
        intent_ids = b_intents.to('cpu').numpy()

        predictions_intents.extend(np.argmax(intent_logits, axis=1))
        true_labels_slots.extend(label_ids)
        true_labels_intents.extend(intent_ids)

    predictions_slots = np.array(predictions_slots)
    true_labels_slots = np.array(true_labels_slots)
    predictions_intents = np.array(predictions_intents)
    true_labels_intents = np.array(true_labels_intents)
    return predictions_slots, true_labels_slots, predictions_intents, true_labels_intents


def get_ouputs_fusion1(model, model_tag, model_class, model_graph, max_len, dataloader: DataLoader, device: torch.device):
    """

    computes for evaluation phase in training

    :param PreTrainedModel model: Bert\RoBERTa for token classification
    :param dataloader: dataloader containing the data
    :param device: torch device
    :return:
    """
    final_layer_is_crf = if_final_layer_is_crf(model.__class__)

    encoder_info_filter = lambda info: info

    # Put the model into evaluation mode
    model.eval()
    model_tag.eval()
    model_class.eval()
    model_graph.eval()

    predictions_slots, true_labels_slots, predictions_intents, true_labels_intents = [], [], [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_intents, lens = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, lens=lens, device=device)

            selected_indexes = [[i + 1 for i in range(length)] for idx, length in enumerate(lens.int())]
            padding_lengths = [max_len - len(tokenized_text) for tokenized_text in b_input_ids]

            selected_indexes = [[padding_lengths[idx] + i + idx * max_len for i in selected_index] for
                                idx, selected_index in enumerate(selected_indexes)]
            selects = torch.tensor(list(itertools.chain.from_iterable(selected_indexes)),
                                           device=device)
            copied_indexes = [[i + idx * max_len for i in range(length)] for idx, length in
                              enumerate(lens.int())]
            copies = torch.tensor(list(itertools.chain.from_iterable(copied_indexes)),
                                          device=device)

            tag_scores, tag_space, encoder_info = model_tag(b_input_ids, b_input_mask, lens, selects, copies, max_len,
                                                                 with_snt_classifier=True)

            class_scores, class_space = model_class(encoder_info_filter(encoder_info))
            tag_space, class_space = model_graph(tag_space, class_space, lens)
            tag_scores = F.log_softmax(tag_space, dim=-1)
            class_scores = F.log_softmax(class_space, dim=-1)
        # Move logits and labels to CPU
        intent_logits = outputs[1][0]
        slot_logits = outputs[1][1]

        if type(intent_logits) == list:
            intent_logits = np.array(intent_logits)
        else:
            intent_logits = intent_logits.detach().cpu().numpy()
        if type(class_scores) == list:
            class_scores = np.array(class_scores)
        else:
            class_scores = class_scores.detach().cpu().numpy()
        # Calculate the accuracy for this batch of test sentences.
        # eval_loss += outputs[0].mean().item()

        if not final_layer_is_crf:
            if type(slot_logits) == list:
                slot_logits = np.array(slot_logits[:,:tag_scores.size(1),:])
            else:
                slot_logits = slot_logits.detach().cpu().numpy()[:,:tag_scores.size(1),:]
            if type(tag_scores) == list:
                tag_scores = np.array(tag_scores)
            else:
                tag_scores = tag_scores.detach().cpu().numpy()

            greater_logits = np.greater_equal(np.amax(tag_scores, axis=2), np.amax(slot_logits, axis=2))
            pred_slots_of_batch = []
            for greater, sample1, sample2 in zip(greater_logits, np.argmax(tag_scores, axis=2), np.argmax(slot_logits, axis=2)):
                pred_slots = []
                i = 0
                for x in greater:
                    if x == True:
                        pred_slots.append(sample1[i])
                    else:
                        pred_slots.append(sample2[i])
                    i += 1  
                pred_slots_of_batch.append(pred_slots)
            predictions_slots.extend(pred_slots_of_batch)
        else:
            # CRf does not need argmax
            predictions_slots.extend(np.array(model.crf.decode(slot_logits)))

        label_ids = b_labels.to('cpu').numpy()
        intent_ids = b_intents.to('cpu').numpy()

        greater_logits = np.greater_equal(np.amax(class_scores, axis=1), np.amax(intent_logits, axis=1))
        pred_intents_of_batch = []
        for greater, sample1, sample2 in zip(greater_logits, np.argmax(class_scores, axis=1), np.argmax(intent_logits, axis=1)):
            if greater == True:
                pred_intents_of_batch.append(sample1)
            else:
                pred_intents_of_batch.append(sample2)
        predictions_intents.extend(pred_intents_of_batch)
        true_labels_slots.extend(label_ids)
        true_labels_intents.extend(intent_ids)

    predictions_slots = np.array(predictions_slots)
    true_labels_slots = np.array(true_labels_slots)
    predictions_intents = np.array(predictions_intents)
    true_labels_intents = np.array(true_labels_intents)
    return predictions_slots, true_labels_slots, predictions_intents, true_labels_intents

def cut_at(sens, tags, cut):
    """
    cuts sentences and tags at a threshold

    :param sens: sentences, list of list of string words
    :param tags: tags, list of list of string tags
    :param int cut: threshold
    :return: new_sentences,new_tags with size <= cut
    """
    out_sens = []
    out_tags = []
    for sen, tag in zip(sens, tags):
        sen_ = sen
        tag_ = tag
        while len(sen_) > cut:
            out_sens.append(sen_[:cut])
            out_tags.append(tag_[:cut])
            sen_ = sen_[cut:]
            tag_ = tag_[cut:]
        if len(sen_) > 0:
            out_sens.append(sen_)
            out_tags.append(tag_)
    return out_sens, out_tags


def tag_and_tokens_to_original_form(label_indices, dict2, starts, true_labels):
    """

    :param label_indices: labels indices to match to new tokens
    :param dict2: a dictionary mapping from indices to labels
    :param starts: starts[i]=1 is position i is start of a new word o.w. 0
    :param true_labels: true label indices
    :return: new_labels, new_true
    """
    new_tokens, new_labels, new_true = [], [], []
    for label_idx, start, true_id in zip(label_indices, starts, true_labels):
        if start == 0:
            pass
        else:
            new_labels.append(dict2[label_idx + 1])

            new_true.append(dict2[true_id + 1])

    return new_labels, new_true


def tokenize_and_pad_text(sentences, tokenizer, need_tokenization, max_len):
    """
    tokenizes and pads the sentence

    :param sentences: list of string, input sentences
    :param tokenizer: Bert|RoBERTa tokenizer
    :param max_len: the outputs will have length maxlen, but sentences with size bigger than maxlen - 2  will be truncated (two for [CLS] and [SEP])
    :return: tokenized_sentences_ids ,lens, tokenized_sentences,original_sentences, starts


    """
    tokenized_sentences = []
    tokenized_sentences_ids = []
    lens = []
    original_sentences = []
    starts = []
    for i in range(len(sentences)):

        tokenized_sentence = []
        orig_sen = []
        if need_tokenization==True:
            sentence = sentences[i].split(' ')
        else:
            sentence = sentences[i]
        start = []
        for word in sentence:
            tokenized_word = tokenizer.tokenize(word)
            if len(tokenized_word) == 1:
                start.append(1)
                if tokenized_word[0] != '[UNK]':
                    orig_sen.append(tokenized_word[0])
                else:
                    orig_sen.append(word)
            elif len(tokenized_word) > 0:
                start.append(1)
                for k in range(len(tokenized_word) - 1):
                    start.append(0)
                if '[UNK]' in tokenized_word:
                    orig_sen.extend([word] * len(tokenized_word))
                else:
                    orig_sen.extend(tokenized_word)
            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
        original_sentences.append(orig_sen)
        starts.append(start)
        if len(tokenized_sentence) > max_len - 2:
            print('Warning : Size', len(tokenized_sentence), ' is bigger than maxlen - 2 , truncating index', i)
            tokenized_sentence = tokenized_sentence[:max_len - 2]

        lens.append(len(tokenized_sentence))
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        tokenized_sentence.extend(["[PAD]"] * (max_len - len(tokenized_sentence)))
        tokenized_sentence_id = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        tokenized_sentences.append(tokenized_sentence)
        tokenized_sentences_ids.append(tokenized_sentence_id)
    return np.array(tokenized_sentences_ids, dtype='long'), np.array(lens,
                                        dtype='int64'), tokenized_sentences, original_sentences, starts


def tokenize_and_pad_text_for_train(sentences, tags, intents, tokenizer, max_len, dict_rev2, inte_rev2):
    """
    tokenizes and pads the sentence to feed to training procedure

    :param sentences: list of string, input sentences
    :param tags: corresponding input tags, will be stretched if word are breaked  by tokenizer
    :param tokenizer: Bert|RoBERTa tokenizer
    :param max_len: the outputs will have length maxlen, but sentences with size bigger than maxlen - 2  will be truncated (two for [CLS] and [SEP])
    :return: tokenized_sentences_ids ,lens, tokenized_sentences,original_sentences, starts
    :param dict_rev2:a dictionary mapping from textual labels to indices
    :return: input_ids, lens, tokenized_sentences, tags_ids, intent_ids, starts
    """
    tokenized_sentences = []
    tokenized_sentences_ids = []
    tokenized_tags_ids = []
    intent_ids = []
    lens = []
    starts = []
    for i in range(len(sentences)):

        tokenized_sentence = []
        tokenized_tag = []
        start = []

        sentence = sentences[i]
        tag_list = tags[i]
        for word, tag in zip(sentence, tag_list):
            # print(word)
            tokenized_word = tokenizer.tokenize(word)

            # Add the tokenized word to the final tokenized word list
            if len(tokenized_word) > 0:
                tokenized_sentence.extend(tokenized_word)
                tokenized_tag.extend([dict_rev2[tag]-1] * len(tokenized_word))
                start.append(1)
                start.extend([0] * (len(tokenized_word) - 1))
        starts.append(start)
        if len(tokenized_sentence) > max_len - 2:
            print('Warning : Size', len(tokenized_sentence), ' is bigger, truncating index', i)
            tokenized_sentence = tokenized_sentence[:max_len - 2]
            tokenized_tag = tokenized_tag[:max_len - 2]
        lens.append(len(tokenized_sentence))
        # print('len is ', lens[-1])
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        tokenized_tag = [0] + tokenized_tag + [0]
        if len(tokenized_tag) != len(tokenized_sentence):
            print('BADD')
            exit(0)
        tokenized_sentence.extend(["[PAD]"] * (max_len - len(tokenized_sentence)))
        tokenized_tag.extend([0] * (max_len - len(tokenized_tag)))
        tokenized_sentence_id = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        tokenized_sentences.append(tokenized_sentence)
        tokenized_sentences_ids.append(tokenized_sentence_id)
        tokenized_tags_ids.append(tokenized_tag)
        intent_ids.append(inte_rev2[intents[i]]-1)
    np_tags = np.array(tokenized_tags_ids, dtype='long')
    np_input = np.array(tokenized_sentences_ids, dtype='long')
    np_intents = np.array(intent_ids, dtype='long')
    return np_input, np.array(lens, dtype='int64'), tokenized_sentences, np_tags, np_intents, starts



def join_bpe_split_tokens(tokens, label_indices, dict2, original_sentence, starts):
    """
    joins splitted tokens using information stores in starts

    :param tokens: tokens given by BERT/RoBERTa tokenizer
    :param label_indices: labels indices to match to new tokens
    :param dict2: a dictionary mapping from indices to labels
    :param original_sentence: original text but splitted the same way tokenizer splits , needed because tokenizer puts some [UNK] sme times
    :param starts: starts[i]=1 is position i is start of a new word o.w. 0
    :return: new_tokens, new_labels, final_tokens
    """

    new_tokens, new_labels = [], []
    final_tokens = []
    for token, label_idx, orig_tok, start in zip(tokens, label_indices, original_sentence, starts):
        if start == 0:
            if token.startswith('##'):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_tokens[-1] = new_tokens[-1] + token
        else:
            new_labels.append(dict2[str(label_idx + 1)])
            new_tokens.append(token)
            final_tokens.append(orig_tok)
    return new_tokens, new_labels, final_tokens


class transformertagger():
    """
    This class helps to run inference on models
    """

    def __init__(self, model_path, model_class, dict2, inte2, device=torch.device("cuda")):
        """

        Initializes the class, loads model params and retains it

        :param model_path: the path to load the saved model, model is saved by tagger_trainer
        :param model_class: class of model
        :param device: torch.device, defaults to torch.device("cuda")
        """
  
        self.model = model_class.from_pretrained(model_path, slot_label_lst=[*dict2.keys()], intent_label_lst=[*inte2.keys()])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.model_name)
        self.device = device
        self.final_layer_is_crf = if_final_layer_is_crf(model_class)
        if self.device.type == "cuda":
            self.model.cuda()
        # Put the model into evaluation mode
        self.model.eval()

    def get_label(self, seqs, need_tokenization, bs=32):
        """
        This method returns labels for sequences (seqs)

        :param seqs: list of strings, input sentences
        :param bs: batch size to use in inference defaults to 32
        :return: final_toks, final_labels, final_toks in tokenized version of input sentences and final_labels is corresponding lables
        """

        input_ids, lens, tokenized, original_sentences, starts = tokenize_and_pad_text(seqs, self.tokenizer, need_tokenization, 65)

        attention_masks = [[i < lens[j] + 2 for i in range(len(ii))] for j, ii in enumerate(input_ids)]
        input_ids = input_ids.astype('int64')

        val_inputs = torch.tensor(input_ids)
        val_masks = torch.tensor(attention_masks)

        valid_data = TensorDataset(val_inputs, val_masks)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        predictions_slots, predictions_intents = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)
            # Move logits and labels to CPU
            intent_logits = outputs[1][0]
            slot_logits = outputs[1][1]



            if type(intent_logits) == list:
                intent_logits = np.array(intent_logits)
            else:
                intent_logits = intent_logits.detach().cpu().numpy()


            if not self.final_layer_is_crf:
                if type(slot_logits) == list:
                    slot_logits = np.array(slot_logits)
                else:
                    slot_logits = slot_logits.detach().cpu().numpy()
                predictions_slots.extend([list(p) for p in np.argmax(slot_logits, axis=2)])
            else:
                # CRf does not need argmax
                predictions_slots.extend(np.array(self.model.crf.decode(slot_logits)))

            predictions_intents.extend(np.argmax(intent_logits, axis=1))

        final_toks = []
        final_labels = []
        for i in range(len(tokenized)):
            toks, lbs, final = join_bpe_split_tokens(tokenized[i][1:lens[i] + 1], predictions_slots[i][1:lens[i] + 1], self.model.config.dict2,
                                                     original_sentences[i], starts[i])
            final_toks.append(toks)
            final_labels.append(lbs)

        final_intents = []
        for i in range(len(predictions_intents)):
            final_intents.append(self.model.config.inte2[str(predictions_intents[i]+1)])

        return final_toks, final_labels, final_intents



models_with_crf = ['BertCRF', 'BertLSTMCRF', 'RobertaCRF', 'RobertaLSTMCRF', 'RobertaMLPCRF', 'BertLSTMIDSF']


def if_final_layer_is_crf(model_class):
    """
    check if model has CRF layer a the end

    :param model_class: class of model
    :return: True or False
    """
    if model_class.__name__ in models_with_crf:
        return True
    else:
        return False



def test_printer(data_path, toks, true_labels, predicted_labels, true_intents, predicted_intents):
    for sample_tokens, sample_labels, sample_predicted_label, sample_intent, sample_predicted_intent in zip(toks, true_labels, predicted_labels, true_intents, predicted_intents):
        sample = ""
        for token,label,pred_label in zip(sample_tokens, sample_labels, sample_predicted_label):
            if label == pred_label:
                sample += token + ":" + label + ":" + pred_label + " "
            else:
                sample += token + ":" +  "\033[0;31;47m"+label + "\033[0m:" +  "\033[0;31;47m"+pred_label + "\033[0m "
        if sample_intent == sample_predicted_intent:
            sample += "<=> " + sample_intent + "<=> " + sample_predicted_intent 
        else:
            sample += "<=> " + "\033[0;31;47m"+sample_intent + "\033[0m <=> " +  "\033[0;31;47m"+sample_predicted_intent + "\033[0m"  
        f = open(data_path, "a")
        f.write(sample+"\n")
        f.close()










