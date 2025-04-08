This repository is forked from https://modelscope.cn/models/iic/nlp_deberta_rex-uninlu_chinese-base/files.

This repository is **NOT** depend on ModelScope, only depending on `transformers==4.33.0`.

Quick start:

```
# Download the model and place it at `./nlp_deberta_rex_uninlu_chinese-base`.
$ pip3 install -r requirements.txt
$ python3 demo.py
# {'output': [[{'type': '组织机构', 'span': '北大西洋议会', 'offset': [0, 6]}], [{'type': '地理位置', 'span': '西班牙', 'offset': [14, 17]}], [{'type': '地理位置', 'span': '巴塞罗那', 'offset': [17, 21]}]]}
```

File structure:

- `nlp_deberta_rex_uninlu_chinese-base` contains model files, downloaded from ModelScope.
- `rex` contains training code for the RexUniNLU model. Under Apache-License 2.0.
- `demo.py` is a demo to run the model.
