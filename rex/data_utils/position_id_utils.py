import torch
from torch import nn
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
from .token_config import TYPE_TOKEN, PREFIX_TOKEN
import re
import numpy as np


def build_position_ids_attn_mask(tokenizer: AutoTokenizer, token_id_list: List[int], token_type_ids: List[int], attention_mask: List[int]) -> Tuple[List[int], List[List[int]]]:
    TYPE_ID = tokenizer.convert_tokens_to_ids(TYPE_TOKEN)
    PREFIX_ID = tokenizer.convert_tokens_to_ids(PREFIX_TOKEN)
    segs = {"text": [], "prefix": [], "cls": [0]}
    assert token_id_list[0] == tokenizer.cls_token_id and token_id_list[1] == PREFIX_ID
    pre_special_id = tokenizer.cls_token_id
    for i, t in enumerate(token_id_list):
        if i == 0:
            continue
        if t == PREFIX_ID:
            new_prefix_seg = {"span": [i], "type": []}
            segs["prefix"].append(new_prefix_seg)
            pre_special_id = t
        elif t == TYPE_ID:
            new_type_seg = {"span": [i]}
            segs["prefix"][-1]["type"].append(new_type_seg)
            pre_special_id = t
        elif t == tokenizer.sep_token_id:
            segs["text"].append(i)
            pre_special_id = t
        else:
            # text token
            if pre_special_id == PREFIX_ID:
                segs["prefix"][-1]["span"].append(i)
            elif pre_special_id == TYPE_ID:
                segs["prefix"][-1]["type"][-1]["span"].append(i)
            else:
                assert pre_special_id == tokenizer.sep_token_id
                segs["text"].append(i)
    
    all_len = len(token_id_list)
    # position ids
    position_ids = [i for i in segs["cls"]]
    cls_len = len(position_ids)
    for prefix_seg in segs["prefix"]:
        prefix_len = len(prefix_seg["span"])
        position_ids += [i for i in range(cls_len, cls_len + prefix_len)]
        for type_seg in prefix_seg["type"]:
            type_len = len(type_seg["span"])
            position_ids += [i for i in range(cls_len + prefix_len, cls_len + prefix_len + type_len)]
    pre_max_position_id = max(position_ids)
    position_ids += [i for i in range(pre_max_position_id + 1, pre_max_position_id + 1 + len(segs["text"]))]
    assert len(position_ids) == all_len

    # attention mask
    attention_mask = np.reshape(np.array(attention_mask), (all_len, 1)) * np.reshape(np.array(attention_mask), (1, all_len))
    # prefix to prefix attention mask
    for i in range(len(segs["prefix"])):
        for j in range(len(segs["prefix"])):
            if i != j:
                si, sj = segs["prefix"][i]["span"][0], segs["prefix"][j]["span"][0]
                ei, ej = segs["prefix"][i]["type"][-1]["span"][-1], segs["prefix"][j]["type"][-1]["span"][-1]
                attention_mask[si: ei + 1, sj: ej + 1] = 0
    # type to type attention mask
    for i in range(len(segs["prefix"])):
        for j in range(len(segs["prefix"][i]["type"])):
            for k in range(len(segs["prefix"][i]["type"])):
                if j != k:
                    # mask type span j to k
                    sj, sk = segs["prefix"][i]["type"][j]["span"][0], segs["prefix"][i]["type"][k]["span"][0]
                    ej, ek = segs["prefix"][i]["type"][j]["span"][-1], segs["prefix"][i]["type"][k]["span"][-1]
                    attention_mask[sj: ej + 1, sk: ek + 1] = 0
    return position_ids, attention_mask.tolist()

