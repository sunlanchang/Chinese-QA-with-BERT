# 中文阅读理解任务
针对CMRC中文阅读理解任务的BERT机器阅读理解模型。
## 文件结构
```
tree -L 1
.
├── LICENSE                         
├── README.md
├── __init__.py
├── cmrc2018_evaluate.py            # 评测模型的准确率
├── modeling.py                     # BERT模型
├── optimization.py                 # 优化算法
├── run_cmrc2018.sh                 # 运行BERT模型bash脚本
├── run_cmrc2018_drcd_baseline.py   # 运行BERT模型python文件
├── squad                           # 存放模型训练后的权重
├── squad_base                      # 模型需要的数据
├── tokenization.py                 # 中文分词
└── uncased_L-2_H-128_A-2           # BERT预训练权重，可从官方BERT GitHub下载
```
## 安装
```bash
tensorflow==1.15
```
## 使用方法
1. 从[BERT-Tity](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip)下载模型，存放在uncased_L-2_H-128_A-2文件夹下.
2. `bash run_cmrc2018.sh`