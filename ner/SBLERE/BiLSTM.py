

from transformers import BertModel
import torch.nn as nn
from BiLSTM_config import parsers


class BertBilstm(nn.Module):
    def __init__(self, class_num,bi=True):
        super().__init__()
        print(parsers().bert_pred)
        self.bert = BertModel.from_pretrained(parsers().bert_pred)

        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(768, 384, batch_first=True, bidirectional=bi)

        if bi:
            self.classifier = nn.Linear(384 * 2, class_num)
        else:
            self.classifier = nn.Linear(384, class_num)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch_index):
        bert_out = self.bert(batch_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]
        out, _ = self.lstm(bert_out0)
        pred = self.classifier(out)
        return pred


