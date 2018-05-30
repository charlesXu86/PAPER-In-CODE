# Scanner Semantic Parser

## Introduction
The input is a sentence, while the output is a tree-structured logical form, for which we use s-expression to represent. For example:
- Input: *Which college did Obama go to?*
- Output: `(and (Type University) (Education BarackObama))`

## Training Data
The training data consites of either sentences paired with logical forms (code in the main folder, e.g. for GeoQuery), or sentences paired with answers (code in the kb folder or table folder, e.g., for WebQA, GraphQA and Spades)

## Domains
Knowledge base (kb\_simple) or table domain (table).
Instructions for generating domain-specific input data will be uploaded.


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
