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
├── run_cmrc2018.sh                 # 运行BERT模型bash脚本（简体中文）
├── run_cmrc2018_drcd_baseline.py   # 运行BERT模型bash脚本（繁体中文）
├── squad                           # 存放模型训练后的权重
├── squad_base                      # 模型需要的数据
├── tokenization.py                 # 中文分词
└── uncased_L-2_H-128_A-2           # BERT预训练权重，可从官方BERT GitHub下载
```