# 中文阅读理解任务
针对CMRC中文阅读理解任务的BERT机器阅读理解模型。
## 文件结构
```
$ tree -L 1
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

`bash run_cmrc2018.sh`中：
```bash
export PATH_TO_BERT=uncased_L-2_H-128_A-2
export DATA_DIR=squad
export MODEL_DIR=/tmp/squad_base/
python run_cmrc2018_drcd_baseline.py \                     # 训练的入口python文件
	--vocab_file=${PATH_TO_BERT}/vocab_zh.txt \            # 指定模型需要的中文词典
	--bert_config_file=${PATH_TO_BERT}/bert_config.json \  # 指定bert的网络结构
	--do_train=True \                                      # 是否训练模型
	--init_checkpoint=${PATH_TO_BERT}/bert_model.ckpt \    # 加载的预训练检查点
	--train_file=${DATA_DIR}/cmrc2018_train.json \         # 训练数据
	--do_predict=True \                                    # 训练结束是否做预测
	--predict_file=${DATA_DIR}/cmrc2018_dev.json \         # 验证集
	--train_batch_size=32 \                                # batch size大小
	--num_train_epochs=50 \                                # epoch
	--max_seq_length=512 \                                 # 输入模型的最长句子
	--doc_stride=128 \
	--learning_rate=3e-5 \                                 # 学习率
	--save_checkpoints_steps=1000 \                        # 每1000个batch保存一次检查点
	--output_dir=${MODEL_DIR} \                            # 保存检查点位置
	--do_lower_case=True \                                 # 字母是否小写
	--use_tpu=False                                        # 是否使用TPU
```
## 参考
- [官方Github BERT](https://github.com/google-research/bert)
- [GitHub Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm)
- [CMRC2018的baseline](https://github.com/ymcui/cmrc2018/tree/master/baseline)