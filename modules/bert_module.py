import os

import torch

from pytorch_pretrained_bert.modeling import BertModel
#from transformers.modeling_bert import BertModel

from torch import nn


class BertModule(nn.Module):
    def __init__(self, bert_model_name, cache_dir="./cache/"):
        super().__init__()

        # Create cache directory if not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.bert_model = BertModel.from_pretrained(
            bert_model_name, cache_dir=cache_dir
        )
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.bert_model.to(self.device)

    def forward(self, token_ids, token_type_ids=None, attention_mask=None):
        encoded_layers, pooled_output = self.bert_model(
            token_ids, token_type_ids, attention_mask
        )
        return encoded_layers, pooled_output


class BertLastCLSModule(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, pooled_output):
        last_hidden = pooled_output
        out = self.dropout(last_hidden)
        return [out]
