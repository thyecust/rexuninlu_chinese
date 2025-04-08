This repository is forked from https://modelscope.cn/models/iic/nlp_deberta_rex-uninlu_chinese-base/files.

This repository is **NOT** depend on ModelScope, only depending on `transformers==4.33.0`.

Quick start:

```bash
# Download the model and place it at `./nlp_deberta_rex_uninlu_chinese-base`.
$ pip3 install -r requirements.txt
$ python3 demo.py
# {'output': [[{'type': '组织机构', 'span': '北大西洋议会', 'offset': [0, 6]}], [{'type': '地理位置', 'span': '西班牙', 'offset': [14, 17]}], [{'type': '地理位置', 'span': '巴塞罗那', 'offset': [17, 21]}]]}
```


## Files

```bash
find . -name "*.py" -type f -not -path "*.env*" -not -path "./nlp*" -exec wc -l {} + | sort -n
     0 ./rex/data_utils/__init__.py
     0 ./rex/__init__.py
     0 ./rex/model/__init__.py
     0 ./rex/Trainer/__init__.py
     3 ./rex/data_utils/token_config.py
    73 ./demo.py
    73 ./rex/data_utils/position_id_utils.py
    80 ./rex/model/model.py
    84 ./rex/Trainer/utils.py
   108 ./rex/arguments.py
   149 ./rex/model/rotary.py
   168 ./rex/main.py
   240 ./rex/model/activations.py
   593 ./rex/data_utils/data_loader.py
   817 ./rex/Trainer/trainer.py
  1133 ./rex/model/utils_mod.py
  3521 total
```

File structure:

- `nlp_deberta_rex_uninlu_chinese-base` contains model files, downloaded from ModelScope.
- `rex` contains training code for the RexUniNLU model. Under Apache-License 2.0.
  - `rex/model` model definition.
  - `rex/main.py` eval/train entrypoint.
  - `rex/data_utils` data loader.
  - `rex/data`/`rex/scripts` datasets/scripts used for eval/train.
  - `rex/Trainer` training code.
- `demo.py` is a demo to run the model.

## Evaluation

Modify the `config.ini`, for example

```diff
diff --git a/rex/config.ini b/rex/config.ini
index 9941494..c92bf10 100644
--- a/rex/config.ini
+++ b/rex/config.ini
@@ -1,13 +1,16 @@
-export data_path=data/re
+export data_path=data/ee
 export metrics=span
 export lr=3e-5
 export batch_size=16
 export grad_acc=1
-export bert_model_dir=../
+export bert_model_dir=../nlp_deberta_rex-uninlu_chinese-base
 export logging_steps=100
 export epochs=10
 export lr_type=linear
 export nproc=1
-export load_checkpoint=../
+export load_checkpoint=../nlp_deberta_rex-uninlu_chinese-base
 export run_name=debug
-export output_dir=../../log
\ No newline at end of file
+export output_dir=../log
+
+export NCCL_P2P_DISABLE=1
+export NCCL_IB_DISABLE=1
\ No newline at end of file
```

Now you can run `cd rex && . config.ini && bash scripts/eval.sh` and check results in `log`.

`log/debug/debug.log`

```log
2025/04/08 14:54:15 - root - INFO - loading tokenizer from ../nlp_deberta_rex-uninlu_chinese-base
2025/04/08 14:54:15 - root - INFO - loading model from ../nlp_deberta_rex-uninlu_chinese-base
2025/04/08 14:54:15 - root - INFO - loading checkpoint from ../nlp_deberta_rex-uninlu_chinese-base
2025/04/08 14:54:16 - root - INFO - Loading data from data/ee/dev.json
2025/04/08 14:54:16 - root - INFO - data loading finished
2025/04/08 14:54:16 - root - INFO - initializing trainer
2025/04/08 14:54:16 - root - INFO - ***** Running Test *****
2025/04/08 14:54:16 - root - INFO -   Num examples = 173
2025/04/08 14:54:16 - root - INFO -   Batch size = 16
2025/04/08 14:54:22 - root - INFO - metric span ...
2025/04/08 14:54:22 - root - INFO - {'test_precision': 0.31578947368404436, 'test_recall': 0.11257035647277437, 'test_f1': 0.16597510369564727, 'test_pred_num': 190, 'test_gold_num': 533, 'test_correct_num': 60, 'test_runtime': 6.3236, 'test_samples_per_second': 27.358, 'test_steps_per_second': 1.74}
2025/04/08 14:54:22 - root - INFO - {'test_precision': 0.31578947368404436, 'test_recall': 0.11257035647277437, 'test_f1': 0.16597510369564727, 'test_pred_num': 190, 'test_gold_num': 533, 'test_correct_num': 60, 'test_runtime': 6.3236, 'test_samples_per_second': 27.358, 'test_steps_per_second': 1.74}
2025/04/08 14:54:22 - root - INFO - Done! Time consumption: 6.323943138122559, AVG RT: 0.036554584613425194
```

`log/debug/test_pred.json`

```json
[]
[[{"type": "公司裁员(事件触发词)", "span": "裁员", "offset": [5, 7]}, {"type": "主体企业", "span": "可口可乐", "offset": [0, 4]}]]
[[{"type": "股权冻结(事件触发词)", "span": "冻结", "offset": [28, 30]}, {"type": "披露时间", "span": "2020年1月18日", "offset": [5, 15]}], [{"type": "股权冻结(事件触发词)", "span": "冻结", "offset": [28, 30]}, {"type": "主体企业", "span": "佛山市中基投资有限公司", "offset": [30, 41]}]]
[[{"type": "投资巨额亏损(事件触发词)", "span": "亏损", "offset": [116, 118]}, {"type": "披露时间", "span": "4月30日", "offset": [69, 74]}], [{"type": "高管离职(事件触发词)", "span": "辞职", "offset": [60, 62]}, {"type": "披露时间", "span": "4月30日", "offset": [69, 74]}], [{"type": "投资巨额亏损(事件触发词)", "span": "亏损", "offset": [116, 118]}, {"type": "主体企业", "span": "软银集团", "offset": [83, 87]}], [{"type": "高管离职(事件触发词)", "span": "辞职", "offset": [60, 62]}, {"type": "主体企业", "span": "WeWork", "offset": [105, 111]}]]
[[{"type": "股权冻结(事件触发词)", "span": "冻结", "offset": [69, 71]}, {"type": "披露时间", "span": "2018年10月11日", "offset": [5, 16]}], [{"type": "股权冻结(事件触发词)", "span": "冻结", "offset": [69, 71]}, {"type": "主体企业", "span": "重庆玖威医疗科技有限公司", "offset": [19, 31]}]]
[]
[[{"type": "高管离职(事件触发词)", "span": "离职", "offset": [14, 16]}, {"type": "主体企业", "span": "西藏珠峰资源股份有限公司", "offset": [65, 77]}]]
[]
...
```