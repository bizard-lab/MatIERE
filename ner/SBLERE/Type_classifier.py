

import torch.nn as nn
from transformers import BertModel
from type_classifier_config import parsers
import torch


class Type_classifier(nn.Module):
    def __init__(self):
        super(Type_classifier, self).__init__()
        self.args = parsers()
        self.device ="cuda:0" if torch.cuda.is_available() else "cpu"
        self.bert = BertModel.from_pretrained(self.args.bert_pred)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 全连接层
        self.linear = nn.Linear(self.args.num_filters, self.args.class_num)

    def forward(self, x):
        input_ids, attention_mask = x[0].to(self.device), x[1].to(self.device)
        hidden_out = self.bert(input_ids, attention_mask=attention_mask,
                               output_hidden_states=False)  # 控制是否输出所有encoder层的结果
        # shape (batch_size, hidden_size)  pooler_output -->  hidden_out[0]
        pred = self.linear(hidden_out.pooler_output)
        # 返回预测结果
        return pred


