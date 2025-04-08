from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments
import json
import os
from datetime import timedelta

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers.utils import (
    ExplicitEnum,
    cached_property,
    get_full_repo_name,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    logging,
)



logger = logging.get_logger()

if is_torch_available():
    import torch
    import torch.distributed as dist

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()

def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


@dataclass()
class DataArguments():
    stride_len: int = field(default=32)
    max_len: int = field(default=512)
    hint_max_len: int = field(default=256, metadata={"help": "The maximum length of the hint tokens."})
    negative_sampling_rate: int = field(default=1, metadata={"help": "The rate of negative sampling."})
    data_path: str = field(default='', metadata={"help": "The path to the data."})
    prefix_string_max_len: int = field(default=125)
    info_type_max_len: int = field(default=125)
    empty_sampling_rate: float = field(default=1, metadata={"help": "Deprecated as the empty sampling always helps a lot. The rate of empty sampling."})
    in_low_res: bool = field(default=False, metadata={"help": "Whether in low resources mode, including few ratio and few shot."})


@dataclass()
class ModelArguments():
    hidden_size: int = field(default=1024, metadata={"help": "The hidden size of the model."})
    num_attention_heads: int = field(default=1, metadata={"help": "The number of attention heads."})
    attention_probs_dropout_prob: float = field(default=0.3, metadata={"help": "The dropout probability for attention."})
    hidden_dropout_prob: float = field(default=0.3)

    def to_json_string(self):
        attrs = self.__dict__
        return json.dumps(attrs)
    
    def to_dict(self):
        return self.__dict__



@dataclass()
class UIEArguments(TrainingArguments):
    bert_model_dir: str = field(default='')
    load_checkpoint: Optional[str] = field(default='')
    test_only: bool = field(default=False, metadata={"help": "Whether to only test model."})
    task_metrics: str = field(default="strict", metadata={"help": "Choices: `strict`, `triplet`, `trigger,argument`, multiple metrics supported, use `,` as delimiter"})
    verbose_debug: bool = field(default=False, metadata={"help": "If set to true, some debug message would be written, which may lead to long log."})
    do_pretrain: bool = field(default=False, metadata={"help": "Whether in pretrain stage."})
    ddp_timeout: int = field(default=12 * 3600, metadata={"help": "ddp timeout seconds"})
    run_name: str = field(default="debug", metadata={"help": "Required to direct the log file path. Example, `relation/conll04`."})
    output_dir: str = field(default='log')



def get_args() -> Tuple[DataArguments, UIEArguments, ModelArguments]:
    parser = HfArgumentParser((DataArguments, UIEArguments, ModelArguments))
    data_args, training_args, model_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    return data_args, training_args, model_args
