# Generic Conditional Tree Generation Tool
A generic conditional tree generator

## Introduction
The input is a sentence, while the output is a tree, for which we use s-expression to represent. For example:

- Input: *Which college did Obama go to?*
- Output: `(and (Type University) (Education BarackObama))`

Additionally the tool supports another type of output, e.g.,

- Output: `and(Type(University) Education(BarackObama))`

To generate the tree, we use a neuralized stack-LSTM decoder, which generates the tree conditioned on the input. Besides, a canonical generation order needs to be specified. Currently, the tool supports two generation orders:

- Pre-order. It generates the tree top down. For example, the above tree is generated with the following node order `[and, Type, University, Education, BarackObama]`

- Post-order. It generates the tree bottom up. For example, the above tree is generated with the following node order `[University, Type, BarackObama, Education, and]`

## Dependencies
* Numpy
* Dynet

## Data
The train/valid/test data is sentence paired with gold expressions. 
```
sentence1 \t expression1 \n
sentence2 \t expression2 \n
...
```

During test, gold expressions are only used for comparison.

## Parameters
```
--dynet-mem          Dynet memory, needs to come first
--word_dim           word embedding size
--nt_dim             (tree) non-terminal embedding size
--ter_dim            (tree) terminal embedding size
--lstm_dim           lstm size shared by all lstms
--nlayers            number of layers, only used for the sentence encoder
--attention          attention type, feedforward or bilinear
--optimizer          optimizer type, sgd or momentum (recommend) or adam or adgrad
--order              canonical generation order of the tree, pre_order or post_order
--dropout            dropout rate
--embedding_file     pre-trained word embedding file
--model_dir          the model path if load from a previous check point
--result_dir         the result path used to store models and test results
--data_dir           the input data dir, which contains train/valid/test in the above format
```

## Citation
If you use the tool in your work, please cite

    @inproceedings{jp2017scanner,
      author = {Jianpeng Cheng and Siva Reddy and Vijay Saraswat and Mirella Lapata},
      booktitle = {ACL},
      title = {Learning Structured Natural Language Representations for Semantic Parsing},
      year = {2017},
    }

## Liscense
The code is liscensed by the University of Edinburgh under the Apache License, Version 2.0.
