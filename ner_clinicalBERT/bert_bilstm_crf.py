import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF


class BertBiLSTMCRF(BertPreTrainedModel):
    def __init__(self, config, hidden_size_lstm=None):
        super(BertBiLSTMCRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_size_lstm = config.hidden_size if hidden_size_lstm is None else hidden_size_lstm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            config.hidden_size,
            self.hidden_size_lstm,
            dropout=config.hidden_dropout_prob,
            bidirectional=True,
            batch_first=True
        )
        self.classifier_lstm = nn.Linear(2 * self.hidden_size_lstm, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output, _ = self.bilstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier_lstm(sequence_output)

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
              attenion_mask_byte = attention_mask.byte()
              loss, logits = -1*self.crf(logits, labels, mask=attention_mask.byte()), self.crf.decode(logits, mask=attention_mask.byte())
            else:
              loss, logits = -1*self.crf(logits, labels), self.crf.decode(logits)
            outputs = loss
        else:
          logits = self.crf.decode(logits, mask=attention_mask.byte())
          outputs = logits

        return outputs  # (loss), scores, (hidden_states), (attentions)