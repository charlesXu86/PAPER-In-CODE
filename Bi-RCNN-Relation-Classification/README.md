Bi-RCNN-Relation-Classification
===============================
此部分代码是对论文《Bidirectional Recurrent Convolutional Neural Network for Relation Classification》的结果复现，词向量使用的是word2vec在Google News上训练出的300维向量，具体结果如下表所示：
| Model    | Input                                    | $F1$  |
| -------- | ---------------------------------------- | ----- |
| $BiRCNN$ | $\overrightarrow{S}$ $and$ $\overleftarrow{S}$ $of$ $all $ $relations$ | 83.25 |
| $BiRCNN$ | $\overrightarrow{S}$ $and$ $\overleftarrow{S}$ $of$ $directed$ $relations,$ $\overrightarrow{S}$ $of$ $Other$ | 82.75 |
| $BRCNN$  | $\overrightarrow{S}$ $and$ $\overleftarrow{S}$ $of$ $directed$ $relations,$ $\overrightarrow{S}$ $of$ $Other$ | 82.87 |

依赖环境
==
```
tensorflow ==1.2   
python >= 3.5
stanford corenlp >= 3.6
```

代码结构说明
======


运行
==
```
python3 pre_process.py (产生模型输入数据)  
python3 bircnn.py  (模型训练)
```

流程
==
>1. 句子解析，实体提取    
>2. 依存解析，最短依存路径提取sdp
>3. 词向量训练（使用了word2vec和GloVe预训练过的词向量），考虑到数据集规模较小，未在此数据集上调优，即在模型训练过程中设置为不变量； 而依存解析关系向量则是在模型的训练过程中一并训练的    
>4. 模型训练，基本结构为RNN + CNN + softmax    

词向量下载链接
==
[https://github.com/xgli/word2vec-api.git](https://github.com/xgli/word2vec-api.git)    该repo中的README.md给出了多份词向量下载地址

依存解析
==
使用stanford corenlp工具包，stanford依存解析总共有多种形式，其中只有basic和treey能够保证树的结构，此处基于basic解析依存关系(树形结构能够保证最短依存路径是唯一的)

最短依存路径生成
==
从entity1和entity2递归向依存树的父亲节点查找，将父亲节点加入当前集合，当两个集合存在交集时结束，此时最短依存路径已经找到，最坏情况下，entity1和entity2查找到根节点时，找到最短依存路径

模型结构
==
![](./figure/model structure.JPG)


仿真参数
==
```
lr = 0.5    
l2 = 10e-5    
keep_prob = 0.5  
batch_size = 128    
word_vec_size = 300    
rel_vec_size = 30    
inverse_other = False    
has_corase_grained = True 
optimizer = AdadeltaOptimizer
```

loss曲线
==

$BiRCNN1$
![](./figure/loss1.PNG)

可以看到在迭代步数达到4k左右时，valid loss达到最小值，之后开始有缓慢的过拟合现象，因此选择此时的模型作为最终训练结果，得到的F1-score为：83.25

> 备注：代码中提供的F1-socre计算方法与官方perl脚本计算结果可能会有零点几个百分点的差异，原因是代码中使用了全精度进行F1计算，而官方perl脚本里的中间变量只保留了两位小数。

$BiRCNN2$
![](./figure/loss2.JPG)
$BRCNN$
![](./figure/loss3.JPG)

