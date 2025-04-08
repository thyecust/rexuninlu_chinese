import os
import json
import torch

from transformers import AutoTokenizer, AutoConfig

from rex.data_utils import data_loader, token_config
from rex.arguments import get_args
from rex.model.model import RexModel
from rex.Trainer.trainer import RexModelTrainer
from rex.Trainer.utils import compute_metrics

class Processor(object):
    def __init__(self, model, preprocessor=None, **kwargs):
        self.model_dir = model
        data_args, training_args, model_args = get_args()
        training_args.bert_model_dir = self.model_dir
        training_args.load_checkpoint = self.model_dir
        # training_args.fp16 = False
        training_args.no_cuda = True
        tokenizer = AutoTokenizer.from_pretrained(training_args.bert_model_dir)
        tokenizer.add_special_tokens({
            "additional_special_tokens": [token_config.PREFIX_TOKEN, token_config.TYPE_TOKEN, token_config.CLASSIFY_TOKEN, token_config.MULTI_CLASSIFY_TOKEN]
        })
        config = AutoConfig.from_pretrained(training_args.bert_model_dir)

        model = RexModel(config, training_args, model_args)
        if training_args.no_cuda:
            model.load_state_dict(torch.load(os.path.join(
                training_args.load_checkpoint, 'pytorch_model.bin'), map_location=torch.device('cpu')), strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(
                training_args.load_checkpoint, 'pytorch_model.bin')), strict=False)

        uie_token_data_loader = data_loader.UIEDataLoader(
            data_args,
            tokenizer,
            data_args.data_path,
            training_args.local_rank,
            training_args.world_size,
            training_args.no_cuda)

        trainer = RexModelTrainer(model, training_args, uie_token_data_loader.get_collate_fn(),
                                  tokenizer=tokenizer,
                                  compute_metrics=compute_metrics
                                  )
        trainer.rex_dl = uie_token_data_loader
        trainer.data_args = data_args
        self.model, self.trainer = model, trainer

    def __call__(self, input, **forward_params):
        """ Provide default implementation using self.model and user can reimplement it
        """
        text = input
        schema = forward_params.pop('schema')
        if type(schema) == str:
            schema = json.loads(schema)

        input_dict = {
            'text': text,
            'schema': schema
        }
        pred_info_list = self.trainer.do_predict(
            self.model, input_dict)
        return {'output': pred_info_list}


if __name__ == "__main__":
    runner = Processor(
        model='nlp_deberta_rex-uninlu_chinese-base', device='gpu')
    print(runner(
        input='[CLASSIFY]因为周围的山水，早已是一派浑莽无际的绿色了。任何事物（候选词）一旦达到某种限度，你就不能再给它(代词)增加什么了。',
        schema={
            '下面的句子中，代词“它”指代的是“事物”吗？是的': None, "下面的句子中，代词“它”指代的是“事物”吗？不是": None,
        }
    ))
    print(runner(
        '[CLASSIFY]有点看不下去了，看作者介绍就觉得挺矫情了，文字也弱了点。后来才发现 大家对这本书评价都很低。亏了。',
        schema={
            '正向情感': None, "负向情感": None
        }
    ))
    print(runner(
        '[MULTICLASSIFY]《格林童话》是德国民间故事集。由德国的雅各格林和威廉格林兄弟根据民间口述材料改写而成。其中的《灰姑娘》、《白雪公主》、《小红帽》、《青蛙王子》等童话故事，已被译成多种文字，在世界各国广为流传，成为各地收集民间故事的范例。',
        schema={
            '民间故事': None, '推理': None, '心理学': None, '历史': None, '传记': None, '外国名著': None, '文化': None, '诗歌': None, '童话': None, '艺术': None, '科幻': None, '小说': None
        }
    ))
