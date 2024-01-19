# 基于对比学习的多任务协调摘要生成模型
## 训练：

```
python main.py --cuda --gpuid 1 2 3 --config cnndm -l 
```

## 使用BART模型生成摘要：

```
python gen_candidate.py --gpuid 3 --src_dir test/test.source --tgt_dir result/test.out_bart
```

## 使用BRIO模型生成摘要：

```
python main.py --cuda --gpuid 2 --config cnndm -e --model_pt model_generation.bin -g
```

## 使用ChatGPT引擎生成摘要：

```
python gen_openai.py
```

## 计算ROUGE分数
```
python cal_rouge.py --hyp test/test.reference --ref result/test.out
```


## 计算BLEU分数

```
python cal_BLEU.py --hyp test/test.reference --ref result/test.out
```

## 计算METEOR分数

```
python cal_METEOR.py --hyp test/test.reference --ref result/test.out
```

