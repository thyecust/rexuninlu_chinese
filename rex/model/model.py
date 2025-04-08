import torch
import logging
import math
from typing import Union, Tuple, Optional
from torch import nn
from copy import deepcopy
from collections import OrderedDict
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, ModuleList
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoModel, DebertaModel, BertModel, RobertaModel

import torch.nn.functional as F
from .rotary import RotaryEmbedding
from .utils_mod import DebertaModel2dAttnMask, DebertaV2Model2dAttnMask


logger = logging.getLogger()


class RexModel(nn.Module):
    def __init__(self, config, args, model_args):
        super().__init__()
        self.args = config
        if config.model_type == "deberta":
            self.plm = DebertaModel2dAttnMask.from_pretrained(args.bert_model_dir)
        elif config.model_type == "deberta-v2":
            self.plm = DebertaV2Model2dAttnMask.from_pretrained(args.bert_model_dir)
        else:
            self.plm = AutoModel.from_pretrained(args.bert_model_dir)
        
        self.plm.resize_token_embeddings(self.plm.embeddings.word_embeddings.weight.shape[0] + 4)

        self.ffnq = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffnk = nn.Linear(config.hidden_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.hidden_size)
        self.bce_loss = BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def circle_loss(self, y_pred, y_true):
        batch_size = y_true.size(0)
        y_true = y_true.view(batch_size, -1)
        y_pred = y_pred.view(batch_size, -1)
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[:, :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def forward(
        self, 
        input_ids, 
        attention_masks, 
        token_type_ids,
        position_ids,
        labels=None
    ):
        if self.args.model_type == "roberta":
            sequence_output = self.plm(input_ids, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            sequence_output = self.plm(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, position_ids=position_ids)[0]
        # (b, l, n)
        ffn_args = (sequence_output,)
        q = self.rotary_emb.rotate_queries_or_keys(self.ffnq(*ffn_args), position_ids=position_ids)
        k = self.rotary_emb.rotate_queries_or_keys(self.ffnk(*ffn_args), position_ids=position_ids)
        logits = torch.bmm(q, k.permute(0, 2, 1))
        if labels is not None:
            loss_func = self.circle_loss
            loss = loss_func(logits, labels)
            return {
                "loss": loss
            }
        else:
            return {
                "prob": self.sigmoid(logits),
                "logits": torch.where(logits > 0)
            }