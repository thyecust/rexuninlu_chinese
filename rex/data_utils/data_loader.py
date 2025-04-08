import os
import torch
import json
import logging
import re
import random
import numpy as np

from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from math import ceil
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    BertTokenizerFast,
    BertTokenizer,
    modeling_utils
)
from datasets import load_dataset
import torch.distributed as dist


from tokenizers import Encoding
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from .token_config import TYPE_TOKEN, PREFIX_TOKEN
from .position_id_utils import build_position_ids_attn_mask

logger = logging.getLogger()

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class UIEDataLoader(object):
    def __init__(self, args, tokenizer, data_dir, rank, world_size, no_cuda):
        self.args = args
        self.stride_len = args.stride_len
        self.max_len = args.max_len
        self.hint_max_len = args.hint_max_len
        self.data_dir = data_dir
        self.rank = rank
        self.world_size = world_size
        self.no_cuda = no_cuda
        self.tokenizer = tokenizer
        self.debug = False
        self.bad_count = 0
        self.build_position_ids_attn_mask = build_position_ids_attn_mask

    def get_brother_type_map(self, schema, brother_type_map, prefix_types):
        if not schema:
            return
        for k in schema:
            brother_type_map[tuple(prefix_types + [k])] += [v for v in schema if v != k]
            self.get_brother_type_map(schema[k], brother_type_map, prefix_types + [k])
    
    def get_schema_by_prefix(self, schema, prefix_tuple):
        if len(prefix_tuple) == 0:
            return schema
        k = prefix_tuple[0][0]
        return self.get_schema_by_prefix(schema[k], prefix_tuple[1:])

    def load_raw_data(self, file_path):
        rank = self.rank
        raw_data = []
        data_fp = f'{self.data_dir}/{file_path}'
        logger.info(f'Loading data from {data_fp}')
        if self.no_cuda:
            with open(data_fp) as f:
                for num_line, line in enumerate(f):
                    raw_sample = json.loads(line)
                    raw_data.append(raw_sample)
        else:
            with open(data_fp) as f:
                for num_line, line in enumerate(f):
                    if num_line % self.world_size != rank:
                        continue
                    raw_sample = json.loads(line)
                    raw_data.append(raw_sample)
        return raw_data
    
    def should_emp_sample(self):
        return True
    
    def choose_x(self, cands, num=1):
        res = []
        while num > 0:
            idx = np.random.randint(0, len(cands))
            t = cands[idx]
            if t in res:
                num += 1
            else:
                res.append(t)
            num -= 1
        return res

    def get_schema_from_info_list(self, info_list, no_neg_sample=False):
        schema = {}
        for info in info_list:
            assert len(info) <= 2
            ent_type = info[0]["type"]
            if ent_type not in schema:
                schema[ent_type] = None
            if len(info) == 2:
                rel_type = info[1]["type"]
                if schema[ent_type] is None:
                    schema[ent_type] = {}
                if rel_type not in schema[ent_type]:
                    schema[ent_type][rel_type] = None
        return schema

    def tokenize_data(self, data_fp, tokenized_data_fp=None):
        logger.info(f'Tokenizing data and dump to {tokenized_data_fp}')
        with open(data_fp, "r") as f:
            with open(f"{tokenized_data_fp}_{self.rank}", "w") as fo:
                for sample_id, line in enumerate(f):
                    if self.rank != -1 and sample_id % self.world_size != self.rank:
                        continue
                    if self.rank == -1 or sample_id % 100000 == self.rank == 0:
                        logger.info(f'Load data number:{sample_id}')
                    raw_sample = json.loads(line)
                    
                    if self.tokenizer.additional_special_tokens[2] in raw_sample['text']:
                        data_type = 'cls'
                        cls_token = self.tokenizer.additional_special_tokens[2]
                        cls_token_id = self.tokenizer.additional_special_tokens_ids[2]
                    elif self.tokenizer.additional_special_tokens[3] in raw_sample['text']:
                        data_type = 'multi_cls'
                        cls_token = self.tokenizer.additional_special_tokens[3]
                        cls_token_id = self.tokenizer.additional_special_tokens_ids[3]
                    else:
                        data_type = 'no_cls'
                        
                    brother_type_map = defaultdict(list)
                    # the types of the same level (like tuple, triple, quadruple)
                    sample_schema = raw_sample["schema"] if "schema" in raw_sample else self.get_schema_from_info_list(raw_sample["info_list"])
                    self.get_brother_type_map(sample_schema, brother_type_map, [])

                    # following are naishan design
                    info_list_by_level = defaultdict(list)
                    for info in raw_sample['info_list']:
                        for i, x in enumerate(info):
                            if info[:i+1] not in info_list_by_level[i]:
                                info_list_by_level[i].append(info[:i+1])
                    # info_list_by_level: 1 level (tuple), 2 level (triple), 3 level (quadruple)
                    if self.debug:
                        pred_info_list = []

                    # cur leven hint map, starts from level 0
                    level_hint_map = defaultdict(dict)
                    next_level_hint_map = defaultdict(dict)
                    # negative sampling, also initial
                    level0types = list(sample_schema.keys())
                    level0_prefix_tuple = ()
                    for k in level0types:
                        level_hint_map[level0_prefix_tuple][k] = []

                    max_level_num = max(len(k) for k in brother_type_map)
                    for level in range(max_level_num):
                        level_info_list = info_list_by_level[level]
                        for info in level_info_list:
                            assert raw_sample['text'][info[-1]['offset'][0]: info[-1]['offset'][1]] == info[-1]['span']
                            prefix_tuple = tuple((x["type"], x["span"], tuple(x["offset"])) for x in info[:-1])
                            ent_type = info[-1]['type']
                            ent_span = {
                                'span': info[-1]['span'],
                                'offset': info[-1]['offset']
                            }
                            if ent_type not in level_hint_map[prefix_tuple]:
                                level_hint_map[prefix_tuple][ent_type] = []
                            level_hint_map[prefix_tuple][ent_type].append(ent_span)

                            # negative sampling for next level
                            next_level_prefix_tuple = tuple((x["type"], x["span"], tuple(x["offset"])) for x in info)
                            _next_level_schema = self.get_schema_by_prefix(sample_schema, next_level_prefix_tuple)
                            if _next_level_schema is not None:
                                for k in _next_level_schema:
                                    if k not in next_level_hint_map[next_level_prefix_tuple]:
                                        next_level_hint_map[next_level_prefix_tuple][k] = []

                        # level hint map: [prefix_tuple, ent_type] -> [ent_span] of level i

                        # do negative sampling when processing cur level infos
                        for prefix_tuple in level_hint_map:
                            for ent_type in level_hint_map[prefix_tuple]:
                                full_type_tuple = tuple([x[0] for x in prefix_tuple] + [ent_type])
                                for brother_type in brother_type_map[full_type_tuple]:
                                    if brother_type not in level_hint_map[prefix_tuple] and random.random() < self.args.negative_sampling_rate:
                                        level_hint_map[prefix_tuple][brother_type] = []
                                break
                        
                        # pprint(level_hint_map)
                        level_hint_char_map, level_hints = self.split_hint_by_level(level_hint_map)
                        text = raw_sample['text']
                        for i, level_hint in enumerate(level_hints):
                            level_split_hint_char_map = level_hint_char_map[i]
                            
                            if data_type == 'no_cls':
                                tokenized_input = self.tokenizer(
                                    level_hint,
                                    text,
                                    truncation="only_second",
                                    max_length=self.max_len,
                                    stride=self.stride_len,
                                    return_token_type_ids=True,
                                    return_overflowing_tokens=True,
                                    return_offsets_mapping=True
                                )
                                for input_ids, token_type_ids, attention_mask, offset_mapping in zip(
                                    tokenized_input['input_ids'], 
                                    tokenized_input['token_type_ids'], 
                                    tokenized_input['attention_mask'], 
                                    tokenized_input['offset_mapping']
                                ):
                                    if sum(token_type_ids) == 0:
                                        # rebuild token_type_ids
                                        token_type_ids = []
                                        pre_token_id = -1
                                        cur_type_id = 0
                                        for t in input_ids:
                                            if pre_token_id == self.tokenizer.sep_token_id and t != self.tokenizer.sep_token_id:
                                                cur_type_id = 1
                                            token_type_ids.append(cur_type_id)
                                            pre_token_id = t

                                    rows, cols, level_split_pred_info_list = self._get_labels(text, input_ids, token_type_ids, offset_mapping, level_hint_map, level_split_hint_char_map)
                                    if self.debug and level == len(info_list_by_level)-1:
                                        pred_info_list += level_split_pred_info_list
                                    sample = {
                                        'id': sample_id,
                                        'num_tokens': len(input_ids),
                                        'input_ids': input_ids,
                                        'attention_masks': attention_mask,
                                        'token_type_ids': token_type_ids,
                                        'rows': rows,
                                        'cols': cols
                                    }
                                    fo.write(json.dumps(sample, ensure_ascii=False)+'\n')
                            
                            else:
                                tokenized_input = self.tokenizer(
                                    level_hint,
                                    text,
                                    truncation="only_second",
                                    max_length=self.max_len,
                                    stride=0,
                                    return_token_type_ids=True,
                                    return_overflowing_tokens=True,
                                    return_offsets_mapping=True
                                )
                                for j, (input_ids, token_type_ids, attention_mask, offset_mapping) in enumerate(zip(
                                    tokenized_input['input_ids'], 
                                    tokenized_input['token_type_ids'], 
                                    tokenized_input['attention_mask'], 
                                    tokenized_input['offset_mapping']
                                )):
                                    if j != 0:
                                        input_text_start_ids = input_ids.index(self.tokenizer.sep_token_id) + 1
                                        input_ids = input_ids[:input_text_start_ids] + [cls_token_id] + input_ids[input_text_start_ids: -2] + [input_ids[-1]]
                                        
                                        text_offset_start_idx = offset_mapping[input_text_start_ids][0]
                                        cls_sp_token_len = len(cls_token)
                                        cls_sp_token_offset = (text_offset_start_idx + (j-1)*cls_sp_token_len, text_offset_start_idx + j * cls_sp_token_len)
                                        text_offset_mapping = offset_mapping[input_text_start_ids: -2]
                                        text_offset_mapping = [cls_sp_token_offset] + [(offset[0] + j * cls_sp_token_len, offset[1] + j * cls_sp_token_len) for offset in text_offset_mapping] + [offset_mapping[-1]]
                                        offset_mapping = offset_mapping[:input_text_start_ids] + text_offset_mapping
                                        
                                        for prefix_tuple in level_hint_map:
                                            for cls_type in level_hint_map[prefix_tuple]:
                                                golden_cls = level_hint_map[prefix_tuple][cls_type]
                                                for g in golden_cls:
                                                    level_hint_map[prefix_tuple][cls_type] = [{'span': cls_token, 'offset': [cls_sp_token_offset[0], cls_sp_token_offset[1]]}]
                                                    
                                                
                                    if sum(token_type_ids) == 0:
                                        # rebuild token_type_ids
                                        token_type_ids = []
                                        pre_token_id = -1
                                        cur_type_id = 0
                                        for t in input_ids:
                                            if pre_token_id == self.tokenizer.sep_token_id and t != self.tokenizer.sep_token_id:
                                                cur_type_id = 1
                                            token_type_ids.append(cur_type_id)
                                            pre_token_id = t
                                    rows, cols, level_split_pred_info_list = self._get_labels(text, input_ids, token_type_ids, offset_mapping, level_hint_map, level_split_hint_char_map)
                                    if self.debug and level == len(info_list_by_level)-1:
                                        pred_info_list += level_split_pred_info_list
                                    sample = {
                                        'id': sample_id,
                                        'num_tokens': len(input_ids),
                                        'input_ids': input_ids,
                                        'attention_masks': attention_mask,
                                        'token_type_ids': token_type_ids,
                                        'rows': rows,
                                        'cols': cols
                                    }
                                    fo.write(json.dumps(sample, ensure_ascii=False)+'\n')

                        if self.debug:
                            a, b, c = compute_corrects(pred_info_list, raw_sample['info_list'], 'info_strict_f1')
                            if not a == b == c:
                                print(num_line)
                                print(a, b, c)
                                for x in sorted(raw_sample['info_list'], key=lambda t: str(t)):
                                    print(x)
                                print('\n')
                                for x in sorted(pred_info_list, key=lambda t: str(t)):
                                    print(x)
                                input()
                        level_hint_map = next_level_hint_map
                        next_level_hint_map = defaultdict(dict)
        logger.warn("rank %d tokenizing done", self.rank)
        if self.world_size > 1:
            dist.barrier()
        if self.rank == 0 or self.rank == -1:
            logger.info('Merging tokenized data from each process !')
            with open(tokenized_data_fp, 'w') as fo:
                for i in range(self.world_size):
                    from_p = f'{tokenized_data_fp}_{i}' if self.rank >= 0 else f'{tokenized_data_fp}_{-1}'
                    with open(from_p) as fi:
                        for line in fi:
                            fo.write(line)
                    os.remove(from_p)
            logger.info('Merge Done and delete temp files!')
        if self.world_size > 1:
            dist.barrier()


    def split_hint_by_level(self, level_hint_map):
        prefix_tuples = list(level_hint_map.keys())
        random.shuffle(prefix_tuples)
        level_hint_char_map = defaultdict(dict)
        level_hints = []
        level_hint = ''
        len_token_level_hint = 2
        for prefix_tuple in prefix_tuples:
            ent_types = list(level_hint_map[prefix_tuple].keys())
            random.shuffle(ent_types)
            ent_types = sorted(ent_types)
            prefix_string = ','.join([f'{x[0]}: {x[1]}' for x in prefix_tuple])
            len_token_prefix_string = len(self.tokenizer(prefix_string)['input_ids']) - 1
            if len_token_prefix_string > self.args.prefix_string_max_len:
                continue
            is_first_ent_type = True
            # for all relations for one entity
            for i, ent_type in enumerate(ent_types):
                len_token_ent_type = len(self.tokenizer(ent_type)['input_ids']) - 1
                if len_token_ent_type > self.args.info_type_max_len:
                    continue
                if (is_first_ent_type and len_token_level_hint + len_token_prefix_string + len_token_ent_type > self.hint_max_len) \
                        or (not is_first_ent_type and len_token_level_hint + len_token_ent_type > self.hint_max_len):
                    if len(level_hint) > 0:
                        level_hints.append(level_hint)
                    level_hint = PREFIX_TOKEN + prefix_string
                    level_hint_char_map[len(level_hints)][(prefix_tuple, ent_type)] = len(level_hint)
                    assert 2 + len_token_prefix_string + len_token_ent_type <= self.hint_max_len
                    level_hint += f'{TYPE_TOKEN}{ent_type}'
                    len_token_level_hint = 2 + len_token_prefix_string + len_token_ent_type
                else:
                    if is_first_ent_type:
                        level_hint += PREFIX_TOKEN + prefix_string
                        level_hint_char_map[len(level_hints)][(prefix_tuple, ent_type)] = len(level_hint)
                        level_hint += f'{TYPE_TOKEN}{ent_type}'
                        len_token_level_hint += len_token_prefix_string + len_token_ent_type
                    else:
                        level_hint_char_map[len(level_hints)][(prefix_tuple, ent_type)] = len(level_hint)
                        level_hint += f'{TYPE_TOKEN}{ent_type}'
                        len_token_level_hint += len_token_ent_type
                is_first_ent_type = False
        if len(level_hint) > 0:
            level_hints.append(level_hint)
        return level_hint_char_map, level_hints
 


    def load_data(self, file_path, output_dir):
        """
        Arguments:
            full_data_fp: Optional[str]. direct path to the raw data.
        """
        data_fp = f'{self.data_dir}/{file_path}'
        tokenized_data_fp = os.path.join(self.data_dir, f'tokenized_{file_path}')
        logger.info(f'Loading data from {data_fp}')
        
        # if not os.path.exists(tokenized_data_fp):
        self.tokenize_data(data_fp, tokenized_data_fp)
        # else:
        #     logger.info(f'{tokenized_data_fp} already exists, Reuse it!')
        tokenized_data = load_dataset("json", data_files=tokenized_data_fp)
        num_data = len(tokenized_data['train'])
        train_dataset = tokenized_data['train']
        # for index in random.sample(range(len(train_dataset)), 3):
        #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        return train_dataset

 
    def _get_labels(self, text, input_ids, token_type_ids, offset_mapping, level_hint_map, level_split_hint_char_map):
        num_tokens = len(input_ids)
        num_hint_tokens = sum([int(x == 0) for x in token_type_ids])
        text_range = [offset_mapping[num_hint_tokens][0], offset_mapping[-2][1]]
        rows = []
        cols = []
        char_index_to_token_index_map = {}
        for i in range(num_hint_tokens, num_tokens):
            offset = offset_mapping[i]
            for j in range(offset[0], offset[1]):
                char_index_to_token_index_map[j] = i
        level_split_hint_token_map = {}
        hint_char_index_to_token_index_map = {}
        for i in range(num_hint_tokens):
            offset = offset_mapping[i]
            hint_char_index_to_token_index_map[offset[0]] = i
        for x in level_split_hint_char_map:
            # x: (prefix_tuple, ent_type)
            level_split_hint_token_map[x] = hint_char_index_to_token_index_map[level_split_hint_char_map[x]]

        for prefix_tuple in level_hint_map:
            for ent_type in level_hint_map[prefix_tuple]:
                if (prefix_tuple, ent_type) in level_split_hint_token_map:
                    hint_token_index = level_split_hint_token_map[(prefix_tuple, ent_type)]
                    entities = level_hint_map[prefix_tuple][ent_type]
                    for e in entities:
                        h, t = e['offset']
                        t -= 1
                        if h >= text_range[0] and t <= text_range[1]:
                            while h not in char_index_to_token_index_map:
                                h += 1
                                if h > len(text):
                                    print('h', e['offset'], e['span'], text[e['offset'][0]: e['offset'][1]])
                                    break
                            while t not in char_index_to_token_index_map:
                                t -= 1
                                if t < 0:
                                    print('t', e['offset'], e['span'], text[e['offset'][0]: e['offset'][1]])
                                    break
                            if h > len(text) or t < 0:
                                continue
                            token_head = char_index_to_token_index_map[h]
                            token_tail = char_index_to_token_index_map[t]
                            # debug
                            if self.debug:
                                inf_char_head = offset_mapping[token_head][0]
                                inf_char_tail = offset_mapping[token_tail][1]
                                if e['offset'][0] != inf_char_head or e['offset'][1] != inf_char_tail:
                                    # print(tokenized_input.tokens)
                                    print(offset_mapping)
                                    print(inf_char_head, inf_char_tail)
                                    print(e['offset'])
                                    tmp = {
                                        'a': text[inf_char_head: inf_char_tail],
                                        'b': text[e['offset'][0]: e['offset'][1]]
                                    }
                                    print(tmp)
                                    self.bad_count += 1
                                    input()
                                assert token_head <= token_tail, (e['offset'], h, t, text, e['span'])
                            rows += [token_head, token_head, hint_token_index]
                            cols += [token_tail, hint_token_index, token_tail]
        if self.debug:
            token_index_hint_map = {v: k for k, v in level_split_hint_token_map.items()}
            hint_head_map = defaultdict(list)
            hint_tail_map = defaultdict(list)
            spans = []
            for i, j in zip(rows, cols):
                if i >= num_hint_tokens and j >= num_hint_tokens:
                    spans.append((i, j))
                if i < num_hint_tokens and j >= num_hint_tokens:
                    if i in token_index_hint_map:
                        x = token_index_hint_map[i]
                        hint_head_map[x].append(j)
                if i >= num_hint_tokens and j < num_hint_tokens:
                    if j in token_index_hint_map:
                        x = token_index_hint_map[j]
                        hint_tail_map[x].append(i)
            level_split_pred_info_list = []
            for (i, j) in spans:
                for x in level_split_hint_token_map:
                    if j in hint_head_map[x] and i in hint_tail_map[x]:
                        prefix_tuple, ent_type = x
                        char_head = offset_mapping[j][0]
                        char_tail = offset_mapping[i][1]
                        info = [{'type': tmp[0], 'span': tmp[1], 'offset': list(tmp[2])} for tmp in prefix_tuple]
                        info += [
                            {
                                'type': ent_type,
                                'span': text[char_head: char_tail],
                                'offset': [char_head, char_tail]
                            }
                        ]
                        level_split_pred_info_list.append(info)
            return rows, cols, level_split_pred_info_list
        return rows, cols, []

    def get_data_loader(self, dataset, batch_size, rank):
        sampler = DistributedSampler(dataset, rank=rank) if not self.debug else SequentialSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._nn_collate_fn,
            drop_last=False
        )
        return data_loader

    def _padding(self, data, max_len, val=0):
        res = []
        for seq in data:
            if isinstance(seq[0], int):
                res.append(seq + [val]*(max_len-len(seq)))
            else:
                seq = seq + [[] for _ in range(max_len - len(seq))]
                res.append([seq[i] + [val] * (max_len - len(seq[i])) for i in range(len(seq))])
        return res
    
    def _padding_pos_ids(self, data, max_len):
        res = []
        for seq in data:
            fina = seq[-1]
            res.append(seq + list(range(fina + 1, fina + 1 + max_len - len(seq))))
        return res

    def _build_labels(self, rows, cols, max_len):
        labels = np.zeros((max_len, max_len))
        for i, j in zip(rows, cols):
            labels[i, j] = 1
        return labels.tolist()

    def get_collate_fn(self):
        def func(batch):
            batch_max_len = max([item['num_tokens'] for item in batch])
            batch_max_len += (8 - batch_max_len%8)%8

            input_ids = torch.tensor(self._padding([item['input_ids'] for item in batch], batch_max_len, self.tokenizer.pad_token_id), dtype=torch.long)
            
            position_ids, attn_mask = [], []
            for sample in batch:
                sample_position_ids, sample_attn_mask = build_position_ids_attn_mask(self.tokenizer, sample["input_ids"], sample["token_type_ids"], sample["attention_masks"])
                position_ids.append(sample_position_ids)
                attn_mask.append(sample_attn_mask)
            position_ids = torch.tensor(self._padding_pos_ids(position_ids, batch_max_len), dtype=torch.long)
            
            attention_masks = torch.tensor(self._padding(attn_mask, batch_max_len), dtype=torch.long)
            token_type_ids = torch.tensor(self._padding([item['token_type_ids'] for item in batch], batch_max_len), dtype=torch.long)
            labels = torch.tensor([self._build_labels(item['rows'], item['cols'], batch_max_len) for item in batch], dtype=torch.float)

            # collated_batch = [input_ids, attention_masks, token_type_ids, labels]
            collated_batch = {
                'input_ids': input_ids,
                'attention_masks': attention_masks,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'labels': labels
            }
            return collated_batch
        return func


# def tokenize_test():
#     from config import parser
#     args = parser.parse_args()
#     args.hint_max_len = 256
#     args.overwrite_cache = True
#     import pathlib
#     curr_path =  str(pathlib.Path(__file__).parent.absolute())
#     args.bert_model_dir = 'hfl/chinese-roberta-wwm-ext'
#     args.data_dir = os.path.join(curr_path, '../../uie-data/chinese/processed_data/EventExtraction/DUEE_FIN_LITE')
#     uie_dl = UIEDataLoader(args)
#     # uie_dl.debug = True
#     train_data = uie_dl.load_data('dev.json', 0)
#     # valid_data = uie_dl.load_data('dev.json')
#     train_dl = uie_dl.get_data_loader(train_data, batch_size=2, rank=0)
#     for token_ids, attention_masks, token_type_ids, labels in train_dl:
#         print('token_ids', token_ids)
#         print('attention_masks', attention_masks)
#         print('token_type_ids', token_type_ids)
#         print('labels', torch.where(labels==1))
#         input()
    # raw_data_map = {
    #     s['id']: s for s in uie_dl.load_raw_data('dev.json')
    # }
    # for a, (b, c, d, e, f, g) in zip(valid_tokenized_data[::1], valid_dl):
    #     print(b.size())
    #     pprint(raw_data_map['-'.join(a['id'].split('-')[:-1])])
    #     for h, x,y in zip(a['hint_tokens']+['']*(len(d[0])-len(a['hint_tokens'])), d[0], e[0]):
    #         print(h, '\t', x.item(), '\t', y.item())
    #     print()
    #     for u,v,w,x,y in zip(a['tokens']+['']*(len(b[0])-len(a['tokens'])), b[0], c[0], f[0], g[0]):
    #         print(u, '\t', (v.item(), w.item()), '\t', (x.item(), y.item()))
    #     input()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # tokenize_test()