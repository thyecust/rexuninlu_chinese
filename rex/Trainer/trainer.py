from transformers import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_utils import speed_metrics, EvalLoopOutput, EvalPrediction, PredictionOutput
from torch.utils.data import DataLoader, Dataset
# from transformers.utils import logging
import logging
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import time
import os
import jsonlines
import math
import numpy as np
from tqdm import tqdm_notebook as tqdm
import copy
import torch.distributed as dist

logger = logging.getLogger()

class RexModelTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        if self.args.verbose_debug and self.state.global_step % 10000 == 0:
            logger.info("*" * 20)
            logger.info("for debug at global step %d", self.state.global_step)
            input_ids = inputs["input_ids"]
            tokenizer = self.tokenizer
            for sample_idx in range(input_ids.shape[0]):
                logger.info("-" * 10)
                token_list = tokenizer.convert_ids_to_tokens(input_ids[sample_idx].tolist())
                sent = tokenizer.convert_tokens_to_string(token_list)
                logger.info("sample %d sent: %s", sample_idx, sent)
                label_ids = inputs["labels"][sample_idx]
                sample_bounds = torch.nonzero(label_ids).tolist()
                for i in range(len(sample_bounds)):
                    logger.info("sample %d bound: %s(%s) -> %s(%s)", sample_idx, token_list[sample_bounds[i][0]], token_list[sample_bounds[i][0] + 1], token_list[sample_bounds[i][1]], token_list[sample_bounds[i][1] + 1])
                break
            logger.info("*" * 20)

        return (loss, outputs) if return_outputs else loss
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if self.args.do_pretrain and metric_key_prefix != "eval":
            logger.warn("On pretrain stage but in evaluate of mode %s not `eval`. Change to `eval`.", metric_key_prefix)

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.my_loop

        output = eval_loop(
            eval_dataset,
            description="Evaluation" if metric_key_prefix=="eval" else "Test",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.args.world_size > 1:
            dist.barrier()
        if self.args.local_rank <= 0:
            # write output to file
            pred_file = os.path.join(self.args.output_dir, f"{metric_key_prefix}_pred.json")
            with jsonlines.open(pred_file, mode="w") as writer:
                writer.write_all(output.predictions)
            gold_file = os.path.join(self.args.output_dir, f"{metric_key_prefix}_gold.json")
            with jsonlines.open(gold_file, mode="w") as writer:
                writer.write_all(output.label_ids)
        if self.args.world_size > 1:
            dist.barrier()


        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        logger.info(str(output.metrics))

        if metric_key_prefix == "eval" and self.state.best_metric is not None:
            cur_v = output.metrics[f"{metric_key_prefix}_{self.args.metric_for_best_model}"]
            if cur_v > self.state.best_metric:
                logger.info("Achieves best performance, %.4f higher than %.4f", cur_v, self.state.best_metric)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self,
        pred_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "pred",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        output = self.my_loop(
            pred_dataset,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            do_pred=True
        )

        if self.args.world_size > 1:
            dist.barrier()
        if self.args.local_rank <= 0:
            # write output to file
            pred_file = os.path.join(self.args.output_dir, f"{metric_key_prefix}_res.json")
            with jsonlines.open(pred_file, mode="w") as writer:
                writer.write_all(output)
        if self.args.world_size > 1:
            dist.barrier()
        return
    
    def my_loop(
        self,
        dataset,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        do_pred: Optional[bool] = False
    ) -> EvalLoopOutput:
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        # self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        # eval_dataset = getattr(dataloader, "dataset", None)
        num_samples = len(dataset)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        all_preds, all_labels = [], []
        if do_pred:
            for step, inputs in enumerate(dataset):
                # Prediction step
                logits = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys, do_pred = True)
                all_preds.append(copy.deepcopy(logits))
                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            if self.args.world_size > 1:
                # merge all predictions and labels to main process
                dist.barrier()
                all_preds_gathered = [None for _ in range(self.args.world_size)]
                dist.all_gather_object(all_preds_gathered, all_preds)
                all_preds = []
                for i in range(sum(len(x) for x in all_preds_gathered)):
                    all_preds.append(all_preds_gathered[i % self.args.world_size][i // self.args.world_size])
                dist.barrier()
            return all_preds
        else:
            for step, inputs in enumerate(dataset):

                # Prediction step
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                all_preds.append(copy.deepcopy(logits))
                all_labels.append(copy.deepcopy(labels))

                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.world_size > 1:
                # merge all predictions and labels to main process
                dist.barrier()
                all_preds_gathered = [None for _ in range(self.args.world_size)]
                all_labels_gathered = [None for _ in range(self.args.world_size)]
                dist.all_gather_object(all_preds_gathered, all_preds)
                dist.all_gather_object(all_labels_gathered, all_labels)
                all_preds, all_labels = [], []

                for i in range(sum(len(x) for x in all_preds_gathered)):
                    all_preds.append(all_preds_gathered[i % self.args.world_size][i // self.args.world_size])
                    all_labels.append(all_labels_gathered[i % self.args.world_size][i // self.args.world_size])
                dist.barrier()

            # Metrics!
            metric_names = [args.task_metrics] if "," not in args.task_metrics else args.task_metrics.split(",")
            if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
                if args.include_inputs_for_metrics:
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs),
                        metric_names
                    )
                else:
                    metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), metric_names)
            else:
                metrics = {}

            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            # metrics = denumpify_detensorize(metrics)

            if all_losses is not None:
                metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        do_pred: Optional[bool] = False
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        pred_info_list = []
        with torch.no_grad():
            with self.compute_loss_context_manager():
                schema = inputs['schema']
                text = inputs['text']
                legal_output_type_list = self.get_legal_output_type_list(inputs)
                self.prompt_loop(model, text, {(): schema}, pred_info_list, legal_output_type_list)
        
        if not pred_info_list and (self.tokenizer.additional_special_tokens[2] in text or self.tokenizer.additional_special_tokens[3] in text):
            with self.compute_loss_context_manager():
                schema = inputs['schema']
                text = inputs['text']
                legal_output_type_list = self.get_legal_output_type_list(inputs)
                self.prompt_loop(model, text, {(): schema}, pred_info_list, legal_output_type_list, True)
        if self.tokenizer.additional_special_tokens[2] in text and pred_info_list:
            # cls
            keys_prob = [sum([item['confidence'] for item in items])/len(items) for items in pred_info_list]
            max_idx = keys_prob.index(max(keys_prob))
            pred_info_list = pred_info_list[max_idx]
            for item in pred_info_list:
                item.pop('confidence')
            pred_info_list = [pred_info_list]
        if self.tokenizer.additional_special_tokens[3] in text and pred_info_list:
            # multi_cls
            keys_prob = [sum([item['confidence'] for item in items])/len(items) for items in pred_info_list]
            keys_idx = [i for i,v in enumerate(keys_prob) if v > 0.9]
            if not keys_idx:
                keys_idx = [keys_prob.index(max(keys_prob))]
            pred_info_list = [items for i, items in enumerate(pred_info_list) if i in keys_idx]
            for items in pred_info_list:
                for item in items:
                    item.pop('confidence')
        if do_pred:
            return pred_info_list
        else:
            return (None, pred_info_list, inputs['info_list'])
    
    def get_legal_output_type_list(self, inputs):
        if 'legal_output_type_list' in inputs:
            schema_list = inputs['legal_output_type_list']
        else:
            schema = inputs['schema']
            def helper(schema, schema_list, prefix):
                if not schema:
                    schema_list.append(prefix)
                    return
                for k in schema:
                    helper(schema[k], schema_list, prefix+[k])
            schema_list = []
            helper(schema, schema_list, [])
        legal_output_type_list = set()
        for x in schema_list:
            legal_output_type_list.add(tuple(x))
        return legal_output_type_list

    def prompt_loop(self, model, text, level_hint_map, pred_info_list, legal_output_type_list, blank=False):
        rex_dl = self.rex_dl
        level_hint_char_map, level_hints = rex_dl.split_hint_by_level(level_hint_map)
        next_level_hint_map = {}
        
        if self.tokenizer.additional_special_tokens[2] in text:
            data_type = 'cls'
            cls_token = self.tokenizer.additional_special_tokens[2]
            cls_token_id = self.tokenizer.additional_special_tokens_ids[2]
        elif self.tokenizer.additional_special_tokens[3] in text:
            data_type = 'multi_cls'
            cls_token = self.tokenizer.additional_special_tokens[3]
            cls_token_id = self.tokenizer.additional_special_tokens_ids[3]
        else:
            data_type = 'no_cls'
        
        pred_info_cands = []
        for i, level_hint in enumerate(level_hints):
            level_split_hint_char_map = level_hint_char_map[i]
            
            if data_type == 'no_cls':

                tokenized_input = self.tokenizer(
                    level_hint,
                    text,
                    truncation="only_second",
                    max_length=rex_dl.max_len,
                    stride=rex_dl.stride_len,
                    return_overflowing_tokens=True,
                    return_token_type_ids=True,
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

                    position_ids, attn_mask = rex_dl.build_position_ids_attn_mask(self.tokenizer, input_ids, token_type_ids, attention_mask)

                    batch_data = [torch.tensor([x], dtype=torch.long) if x is not None else None for x in [input_ids, attn_mask, token_type_ids, position_ids]]
                    batch_data = self._prepare_inputs(batch_data)
                    _, rows, cols = model(*batch_data)["logits"]
                    
                    rows = rows.tolist()
                    cols = cols.tolist()
                    next_level_hint_map_, new_preds = self.infer_info_from_prediction(
                        rows, cols,
                        input_ids,
                        token_type_ids,
                        text,
                        offset_mapping,
                        level_split_hint_char_map,
                        level_hint_map,
                        legal_output_type_list
                    )
                    next_level_hint_map.update(next_level_hint_map_)
                    pred_info_list += new_preds
            else:
                pred_info = defaultdict(list)
                tokenized_input = self.tokenizer(
                    level_hint,
                    text,
                    truncation="only_second",
                    max_length=rex_dl.max_len,
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
                        text_offset_start_idx = offset_mapping[input_text_start_ids][0]
                        
                        input_ids = input_ids[:input_text_start_ids] + [cls_token_id] + input_ids[input_text_start_ids: -2] + [input_ids[-1]]
                        text = text[:text_offset_start_idx] + cls_token + text[text_offset_start_idx:]
                        cls_sp_token_len = len(cls_token)
                            
                        cls_sp_token_offset = (text_offset_start_idx + (j-1)*cls_sp_token_len, text_offset_start_idx + j * cls_sp_token_len)
                        text_offset_mapping = offset_mapping[input_text_start_ids: -2]
                        text_offset_mapping = [cls_sp_token_offset] + [(offset[0] + j * cls_sp_token_len, offset[1] + j * cls_sp_token_len) for offset in text_offset_mapping] + [offset_mapping[-1]]
                        offset_mapping = offset_mapping[:input_text_start_ids] + text_offset_mapping
                        
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
                    
                    position_ids, attn_mask = rex_dl.build_position_ids_attn_mask(self.tokenizer, input_ids, token_type_ids, attention_mask)
                    batch_data = [torch.tensor([x], dtype=torch.long) if x is not None else None for x in [input_ids, attn_mask, token_type_ids, position_ids]]
                    batch_data = self._prepare_inputs(batch_data)
                    res = model(*batch_data)
                    
                    probs = res['prob']
                    _, rows, cols = res["logits"]
                    
                    rows = rows.tolist()
                    cols = cols.tolist()
                    decay_rate = 1 ** j
                    if blank:
                        next_level_hint_map_ = self.infer_cls_info_from_prediction_for_blank(
                        rows, cols, probs, pred_info, decay_rate, cls_token,
                        input_ids,
                        token_type_ids,
                        text,
                        offset_mapping,
                        level_split_hint_char_map,
                        level_hint_map,
                        legal_output_type_list
                    )
                    else:
                        next_level_hint_map_ = self.infer_cls_info_from_prediction(
                            rows, cols, probs, pred_info, decay_rate, cls_token,
                            input_ids,
                            token_type_ids,
                            text,
                            offset_mapping,
                            level_split_hint_char_map,
                            level_hint_map,
                            legal_output_type_list
                        )
                    next_level_hint_map.update(next_level_hint_map_)
                
                if data_type == 'cls':
                    if len(pred_info) == 1:
                        key = list(pred_info.keys())[0]
                        pred_info_cands.append((key, [item['confidence'] for item in pred_info[key][0]]))
                    elif len(pred_info) > 1:
                        prob_sum = {}
                        for keys in pred_info:
                            keys_prob = [[item['confidence'] for item in items] for items in pred_info[keys]]
                            prob_sum[keys] = [sum(col)/len(col) for col in zip(*keys_prob)]
                        prob_sum = sorted(prob_sum.items(), key=lambda x: sum(x[1]), reverse=True)
                        pred_info_cands.append((prob_sum[0][0], prob_sum[0][1]))
                elif data_type == 'multi_cls':
                    for keys in pred_info:
                        keys_list = keys.split('[INFOKEY]')
                        info_list = []
                        for i, key in enumerate(keys_list):
                            info_list.append({'type': key, 'span': cls_token, 'offset': [0, len(cls_token)], 'confidence': pred_info[keys][0][i]['confidence']})
                        pred_info_list.append(info_list)
        if data_type == 'cls' and pred_info_cands:
            pred_info_cands = sorted(pred_info_cands, key=lambda x: sum(x[1]), reverse=True)
            keys = pred_info_cands[0][0].split('[INFOKEY]')
            info_list = []
            for i, key in enumerate(keys):
                info_list.append({'type': key, 'span': cls_token, 'offset': [0, len(cls_token)], 'confidence': pred_info_cands[0][1][i]})
            pred_info_list.append(info_list)
                # if len(pred_info) == 1:
                #     pred_info_list.append([{'type': list(pred_info.keys())[0], 'span': cls_token, 'offset': [0, len(cls_token)]}])
                # elif len(pred_info) > 1:
                #     count_dict = {}
                #     for key in pred_info:
                #         count_dict[key] = len(pred_info[key])
                #     count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
                #     if count_dict[0][1] > count_dict[1][1]:
                #         pred_info_list.append([{'type': count_dict[0][0], 'span': cls_token, 'offset': [0, len(cls_token)]}])
                #     elif count_dict[0][1] == count_dict[1][1]:
                #         k = 1
                #         prob_sum = {count_dict[0][0]: sum([item['confidence'] for item in items]) for items in pred_info[count_dict[0][0]]}
                #         while k < len(count_dict) and count_dict[k][1] == count_dict[0][1]:
                #             prob_sum[count_dict[k][0]] = [sum([item['confidence'] for item in items]) for items in pred_info[count_dict[k][0]]][0]
                #             k += 1
                #         prob_sum = sorted(prob_sum.items(), key=lambda x: x[1], reverse=True)
                #         pred_info_list.append([{'type': prob_sum[0][0], 'span': cls_token, 'offset': [0, len(cls_token)]}])
                    
                    # for item in count_dict:
                    #     pred_info_list.append([{'type': item[0], 'span': cls_token, 'offset': [0, len(cls_token)]}])
                    
        if len(next_level_hint_map) == 0:
            return
        self.prompt_loop(model, text, next_level_hint_map, pred_info_list, legal_output_type_list, blank)


    def infer_info_from_prediction(
        self, 
        rows, cols,
        input_ids: List, 
        token_type_ids: List,
        text: str,
        offset_mapping,
        level_split_hint_char_map,
        level_hint_map,
        legal_output_type_list
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        next_level_hint_map = {}
        pred_info_list = []
        num_tokens = len(input_ids)
        num_hint_tokens = sum([int(x == 0) for x in token_type_ids])
        # if self.debug:
        #     print('num_hint_tokens', num_hint_tokens)
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
            level_split_hint_token_map[x] = hint_char_index_to_token_index_map[level_split_hint_char_map[x]]
        token_index_hint_map = {v: k for k, v in level_split_hint_token_map.items()}
        # if self.debug:
        #     print('token_index_hint_map', token_index_hint_map)
        hint_head_map = defaultdict(list)
        hint_tail_map = defaultdict(list)
        spans = []
        for j, i in zip(rows, cols):
            if i >= num_hint_tokens and j >= num_hint_tokens and j <= i:
                spans.append((i, j))
            if i < num_hint_tokens and j >= num_hint_tokens:
                if i in token_index_hint_map:
                    x = token_index_hint_map[i]
                    hint_head_map[x].append(j)
            if i >= num_hint_tokens and j < num_hint_tokens:
                if j in token_index_hint_map:
                    x = token_index_hint_map[j]
                    hint_tail_map[x].append(i)
        # if self.debug:
        #     print('spans', spans)
        #     for (i, j) in spans:
        #         char_head = offset_mapping[j][0]
        #         char_tail = offset_mapping[i][1]
        #         print(text[char_head: char_tail])
        #     print('hint_head_map', hint_head_map)
        #     print('hint_tail_map', hint_tail_map)
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for (i, j) in spans:
            for x in level_split_hint_token_map:
                if j in hint_head_map[x] and i in hint_tail_map[x]:
                    try:
                        prefix_tuple, ent_type = x
                        char_head = offset_mapping[j][0]
                        char_tail = offset_mapping[i][1]
                        while text[char_head] == ' ':
                            char_head += 1
                        if level_hint_map[prefix_tuple][ent_type]:
                            key = prefix_tuple+((ent_type, text[char_head: char_tail], tuple([char_head, char_tail])),)
                            next_level_hint_map[key] = level_hint_map[prefix_tuple][ent_type]
                        info = [{'type': tmp[0].strip(), 'span': tmp[1], 'offset': list(tmp[2])} for tmp in prefix_tuple]
                        info += [
                            {
                                'type': ent_type,
                                'span': text[char_head: char_tail],
                                'offset': [char_head, char_tail]
                            }
                        ]
                        if tuple([tmp['type'] for tmp in info]) in legal_output_type_list:
                            pred_info_list.append(info)
                    except:
                        continue
        return next_level_hint_map, pred_info_list
    
    def infer_cls_info_from_prediction(
        self, 
        rows, cols, probs, pred_info, decay_rate, cls_token,
        input_ids: List, 
        token_type_ids: List,
        text: str,
        offset_mapping,
        level_split_hint_char_map,
        level_hint_map,
        legal_output_type_list
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        next_level_hint_map = {}
        pred_info_list = []
        num_tokens = len(input_ids)
        num_hint_tokens = sum([int(x == 0) for x in token_type_ids])
        # if self.debug:
        #     print('num_hint_tokens', num_hint_tokens)
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
            level_split_hint_token_map[x] = hint_char_index_to_token_index_map[level_split_hint_char_map[x]]
        token_index_hint_map = {v: k for k, v in level_split_hint_token_map.items()}

        hint_head_map = defaultdict(list)
        hint_tail_map = defaultdict(list)
        hint_head_probs = defaultdict(dict)
        hint_tail_probs = defaultdict(dict)
        spans = []
        for j, i in zip(rows, cols):
            if i >= num_hint_tokens and j >= num_hint_tokens and j <= i:
                spans.append((i, j, probs[0, i, j]))
            if i < num_hint_tokens and j >= num_hint_tokens:
                if i in token_index_hint_map:
                    x = token_index_hint_map[i]
                    hint_head_map[x].append(j)
                    hint_head_probs[x][j] = probs[0, i, j]
            if i >= num_hint_tokens and j < num_hint_tokens:
                if j in token_index_hint_map:
                    x = token_index_hint_map[j]
                    hint_tail_map[x].append(i)
                    hint_tail_probs[x][i] = probs[0, i, j]

        token_list = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for (i, j, span_prob) in spans:
            for x in level_split_hint_token_map:
                if j in hint_head_map[x] and i in hint_tail_map[x]:
                    try:
                        prefix_tuple, ent_type = x
                        char_head = offset_mapping[j][0]
                        char_tail = offset_mapping[i][1]
                        head_prob = hint_head_probs[x][j]
                        tail_prob = hint_tail_probs[x][i]
                        while text[char_head] == ' ':
                            char_head += 1
                        if level_hint_map[prefix_tuple][ent_type]:
                            key = prefix_tuple+((ent_type, text[char_head: char_tail], tuple([char_head, char_tail]), round((decay_rate * (span_prob + head_prob + tail_prob)/3).item(), 4)),)
                            next_level_hint_map[key] = level_hint_map[prefix_tuple][ent_type]
                        info_key = [tmp[0].strip() for tmp in prefix_tuple if tmp[1] == cls_token]
                        info_value = [{'type': tmp[0].strip(), 'span': tmp[1], 'offset': list(tmp[2]), 'confidence': tmp[3]} for tmp in prefix_tuple if tmp[1] == cls_token]
                        info_key += [
                            ent_type
                        ]
                        info_key = '[INFOKEY]'.join(info_key)
                        info_value += [
                            {
                                'type': ent_type,
                                'span': text[char_head: char_tail],
                                'offset': [char_head, char_tail],
                                'confidence': round((decay_rate * (span_prob + head_prob + tail_prob)/3).item(), 4)
                            }
                        ]
                        if tuple([tmp['type'] for tmp in info_value]) in legal_output_type_list and text[char_head: char_tail] == cls_token:
                            pred_info[info_key].append(info_value)
                    except:
                        continue
        return next_level_hint_map
    
    def infer_cls_info_from_prediction_for_blank(
        self, 
        rows, cols, probs, pred_info, decay_rate, cls_token,
        input_ids: List, 
        token_type_ids: List,
        text: str,
        offset_mapping,
        level_split_hint_char_map,
        level_hint_map,
        legal_output_type_list
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        next_level_hint_map = {}
        pred_info_list = []
        num_tokens = len(input_ids)
        num_hint_tokens = sum([int(x == 0) for x in token_type_ids])
        # if self.debug:
        #     print('num_hint_tokens', num_hint_tokens)
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
            level_split_hint_token_map[x] = hint_char_index_to_token_index_map[level_split_hint_char_map[x]]
        token_index_hint_map = {v: k for k, v in level_split_hint_token_map.items()}

        hint_head_map = defaultdict(list)
        hint_tail_map = defaultdict(list)
        hint_head_probs = defaultdict(dict)
        hint_tail_probs = defaultdict(dict)           
        
        types_pos = [i for i, x in enumerate(input_ids) if x == self.tokenizer.additional_special_tokens_ids[1]]
        num_hint_tokens_list = [num_hint_tokens] * len(types_pos)
        rows = [num_hint_tokens] * len(types_pos) + types_pos
        cols = types_pos + [num_hint_tokens] * len(types_pos)
        for j, i in zip(rows, cols):
            if i < num_hint_tokens and j >= num_hint_tokens:
                if i in token_index_hint_map:
                    x = token_index_hint_map[i]
                    hint_head_map[x].append(j)
                    hint_head_probs[x][j] = probs[0, i, j]
            if i >= num_hint_tokens and j < num_hint_tokens:
                if j in token_index_hint_map:
                    x = token_index_hint_map[j]
                    hint_tail_map[x].append(i)
                    hint_tail_probs[x][i] = probs[0, i, j]

        i, j, span_prob = num_hint_tokens, num_hint_tokens, probs[0, num_hint_tokens, num_hint_tokens]
        for x in level_split_hint_token_map:
            if j in hint_head_map[x] and i in hint_tail_map[x]:
                try:
                    prefix_tuple, ent_type = x
                    char_head = offset_mapping[j][0]
                    char_tail = offset_mapping[i][1]
                    head_prob = hint_head_probs[x][j]
                    tail_prob = hint_tail_probs[x][i]
                    while text[char_head] == ' ':
                        char_head += 1
                    if level_hint_map[prefix_tuple][ent_type]:
                        key = prefix_tuple+((ent_type, text[char_head: char_tail], tuple([char_head, char_tail]), round((decay_rate * (span_prob + head_prob + tail_prob)/3).item(), 4)),)
                        next_level_hint_map[key] = level_hint_map[prefix_tuple][ent_type]
                    info_key = [tmp[0].strip() for tmp in prefix_tuple if tmp[1] == cls_token]
                    info_value = [{'type': tmp[0].strip(), 'span': tmp[1], 'offset': list(tmp[2]), 'confidence': tmp[3]} for tmp in prefix_tuple if tmp[1] == cls_token]
                    info_key += [
                        ent_type
                    ]
                    info_key = '[INFOKEY]'.join(info_key)
                    info_value += [
                        {
                            'type': ent_type,
                            'span': text[char_head: char_tail],
                            'offset': [char_head, char_tail],
                            'confidence': round((decay_rate * (span_prob + head_prob + tail_prob)/3).item(), 4)
                        }
                    ]
                    if tuple([tmp['type'] for tmp in info_value]) in legal_output_type_list and text[char_head: char_tail] == cls_token:
                        pred_info[info_key].append(info_value)
                except:
                    continue
            
        return next_level_hint_map