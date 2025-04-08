import torch
import numpy as np
import json
import copy
from typing import List, Dict, Any, Tuple, Optional
from transformers.trainer_utils import EvalPrediction
import logging


logger = logging.getLogger()

def compute_metrics(predictions: EvalPrediction, metric_names: List):
    y_pred = predictions.predictions
    y_true = predictions.label_ids
    metric_dict = {}
    for metric_name in metric_names:
        metric_name = metric_name.strip()
        logger.info("metric %s ...", metric_name)
        pred_num, gold_num, correct_num = 0, 0, 0
        for preds, golds in zip(copy.copy(y_pred), copy.copy(y_true)):
            post_preds = []
            for pred in preds:
                x = preprocess_info_by_metric(pred, metric_name)
                if len(x) > 0:
                    post_preds.append(x)

            post_golds = []
            for gold in golds:
                x = preprocess_info_by_metric(gold, metric_name)
                if len(x) > 0:
                    post_golds.append(x)
            
            pred_str_json = set([json.dumps(x, sort_keys=True) for x in post_preds])
            gold_str_json = set([json.dumps(x, sort_keys=True) for x in post_golds])
            pred_num += len(pred_str_json)
            gold_num += len(gold_str_json)

            correct = [x for x in gold_str_json if x in pred_str_json]
            correct_num += len(correct)
        
        eps = 1e-10
        precision = correct_num / (pred_num + eps)
        recall = correct_num / (gold_num + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        prefix = "" if len(metric_names) == 1 else metric_name + "_"
        metric_dict.update({
            f"{prefix}precision": precision,
            f"{prefix}recall": recall,
            f'{prefix}f1': f1,
            f'{prefix}pred_num': pred_num,
            f'{prefix}gold_num': gold_num,
            f'{prefix}correct_num': correct_num,
        })
    if len(metric_names) > 1:
        metric_dict["f1"] = sum([metric_dict[f"{name}_f1"] for name in metric_names]) / len(metric_names)
    return metric_dict

def preprocess_info_by_metric(info, metric_name="strict"):
    if len(info) == 0:
        return []
    new_info = copy.deepcopy(info)
    if metric_name == "strict":
        return new_info
    elif metric_name == "span":
        for i in range(len(new_info)):
            new_info[i].pop("offset")
        return new_info
    elif metric_name == "triplet":
        if len(new_info) != 2:
            return []
        new_info[1]["type"] = new_info[1]["type"].split("(")[0].strip()
        new_info[0].pop("offset")
        new_info[1].pop("offset")
        return new_info
    elif metric_name == "trigger":
        return new_info[0]
    elif metric_name == "argument":
        if len(new_info) == 1:
            return []
        assert len(new_info) == 2
        new_info[0].pop("offset")
        new_info[0].pop("span")
        return new_info
    raise ValueError("Invalid metric %s" % metric_name)
