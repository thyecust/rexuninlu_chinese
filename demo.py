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
        pred_info_list = self.trainer.prediction_step(
            self.model, input_dict, prediction_loss_only=False, do_pred=True)
        return {'output': pred_info_list}


if __name__ == "__main__":
    runner = Processor(
        model='nlp_deberta_rex-uninlu_chinese-base', device='gpu')
    output = runner("北大西洋议会春季会议26日在西班牙巴塞罗那闭幕。", schema={
                    "人物": None, "地理位置": None, "组织机构": None})
    print(output)
