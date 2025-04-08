import datetime
import json
import os
import shutil
import numpy as np
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig, set_seed
import jsonlines
import torch
import torch.distributed as dist

from time import time


def main():
    from data_utils import data_loader, token_config
    from arguments import get_args, DataArguments, UIEArguments
    from model.model import RexModel
    from Trainer.trainer import RexModelTrainer
    from Trainer.utils import compute_metrics
    from modelscope.hub.check_model import check_local_model_is_latest
    from modelscope.utils.constant import Invoke, ThirdParty

    check_local_model_is_latest(
        '../',
        user_agent={
            Invoke.KEY: Invoke.LOCAL_TRAINER,
            ThirdParty.KEY: 'damo/nlp_deberta_rex-uninlu_chinese-base'
        })
    data_args, training_args, model_args = get_args()
    
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)

    if training_args.world_size > 1:
        dist.barrier()
        output_dirs = [None for _ in range(training_args.world_size)]
        dist.all_gather_object(output_dirs, training_args.output_dir)
        training_args.output_dir = output_dirs[0]

    if training_args.local_rank <= 0:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.world_size > 1:
        dist.barrier()
        
    logging.basicConfig(
            format=f"%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(f"{training_args.output_dir}/debug.log"),
                logging.StreamHandler()
            ]
        )
    logger = logging.getLogger()
    if training_args.local_rank != 0 and training_args.local_rank != -1:
        logger.setLevel(logging.WARNING)

    logger.info("output dir for rank %d: %s" % (training_args.local_rank, training_args.output_dir))

    set_seed(training_args.seed)

    
    if (training_args.verbose_debug):
        logger.info("training args:")
        logger.info("\n".join(f"{k}: {v}" for k, v in sorted(dict(vars(training_args)).items())))
        logger.info("data args:")
        logger.info("\n".join(f"{k}: {v}" for k, v in sorted(dict(vars(data_args)).items())))
        logger.info("model args:")
        logger.info("\n".join(f"{k}: {v}" for k, v in sorted(dict(vars(model_args)).items())))
    
    logger.info('loading tokenizer from %s' % training_args.bert_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(training_args.bert_model_dir)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [token_config.PREFIX_TOKEN, token_config.TYPE_TOKEN, token_config.CLASSIFY_TOKEN, token_config.MULTI_CLASSIFY_TOKEN]
    })
    if training_args.world_size > 1:
        dist.barrier()

    logger.info('loading model from %s' % training_args.bert_model_dir)
    config = AutoConfig.from_pretrained(training_args.bert_model_dir)
    # set seed again to asure initializing parameters are the same for each device
    if training_args.world_size > 1:
        dist.barrier()
    set_seed(training_args.seed)
    model = RexModel(config, training_args, model_args)
    if training_args.load_checkpoint != "":
        logger.info('loading checkpoint from %s', training_args.load_checkpoint)
        model.load_state_dict(torch.load(os.path.join(training_args.load_checkpoint, 'pytorch_model.bin')), strict=False)

    if training_args.world_size > 1:
        dist.barrier()

    uie_token_data_loader = data_loader.UIEDataLoader(
        data_args, 
        tokenizer, 
        data_args.data_path, 
        training_args.local_rank, 
        training_args.world_size,
        training_args.no_cuda)
    if training_args.do_train:
        # TODO data change

        train_dataset = uie_token_data_loader.load_data("train.json", training_args.output_dir)
        eval_dataset = uie_token_data_loader.load_raw_data("dev.json")

        logger.info('data loading finished')

        if training_args.world_size > 1:
            dist.barrier()

        logger.info('initializing trainer')
        trainer = RexModelTrainer(model, training_args, uie_token_data_loader.get_collate_fn(),
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.rex_dl = uie_token_data_loader

        trainer.data_args = data_args

        train_result = trainer.train(training_args.resume_from_checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_train and training_args.do_eval:
        # test
        logger.info(f"{trainer.evaluate(eval_dataset, metric_key_prefix='test')}")
    elif training_args.do_eval:
        eval_dataset = uie_token_data_loader.load_raw_data("dev.json")
        logger.info('data loading finished')
        if training_args.world_size > 1:
            dist.barrier()
        logger.info('initializing trainer')
        
        trainer = RexModelTrainer(model, training_args, uie_token_data_loader.get_collate_fn(),
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        start_time = time()
        trainer.rex_dl = uie_token_data_loader
        trainer.data_args = data_args
        logger.info(f"{trainer.evaluate(eval_dataset, metric_key_prefix='test')}")
        end_time = time()
        logger.info(f'Done! Time consumption: {end_time-start_time}, AVG RT: {(end_time-start_time)/len(eval_dataset)}')
    elif training_args.do_predict:
        pred_dataset = uie_token_data_loader.load_raw_data("dev.json")
        logger.info('data loading finished')
        if training_args.world_size > 1:
            dist.barrier()
        logger.info('initializing trainer')
        
        trainer = RexModelTrainer(model, training_args, uie_token_data_loader.get_collate_fn(),
            eval_dataset=pred_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        start_time = time()
        trainer.rex_dl = uie_token_data_loader
        trainer.data_args = data_args
        logger.info(f"{trainer.predict(pred_dataset, metric_key_prefix='pred')}")
        end_time = time()
        logger.info(f'Done! Time consumption: {end_time-start_time}, AVG RT: {(end_time-start_time)/len(pred_dataset)}')


if __name__ == "__main__":
    main()

