from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
import torch
from torch import nn
from models.transformer.ID_classifier import SlotClassifier, IntentClassifier

""" Using Bert with LSTM and CRF """


class BertIDSF(BertPreTrainedModel):
    def __init__(self, config, intent_label_lst, slot_label_lst, n_layers=1):
        super().__init__(config)

        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.n_layers = n_layers
        self.bidirectional = False
        if hasattr(config, 'bi'):
            if config.bi:
                self.bidirectional = True
        config.lstm_size = self.config.hidden_size
        if self.bidirectional:
            config.lstm_size = int(config.lstm_size / 2)
        self.final_lstm = nn.LSTM(config.hidden_size, config.lstm_size, self.n_layers, bidirectional=self.bidirectional)
        # self.final_gru = nn.GRU(config.hidden_size, config.lstm_size, self.n_layers, bidirectional=self.bidirectional)
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.use_crf = config.use_crf

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            intents=None,
            output_attentions=True,
            lens=None,
            device=None
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # lstm_out, (hn, cn) = self.final_lstm(sequence_output)
        # gru_out, (hn, cn) = self.final_gru(sequence_output)

        intent_logits = self.intent_classifier(sequence_output[:, 0, :])
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # Intent Softmax
        if intents is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intents.view(-1))
            total_loss += 0.1*intent_loss

        # Slot Softmax
        if labels is not None:
            if self.use_crf:
                slot_loss = self.crf(slot_logits, labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), labels.view(-1))
            total_loss += 0.9*slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs