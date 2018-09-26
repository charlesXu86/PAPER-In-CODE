## Datasets

To run single task models on SNLI, Quora, Multi-NLI and AllNli model, you need to download the datasets. If you download the data from original sources, you can preprocess them using the [script.bash](https://github.com/wasiahmad/universal_sentence_encoder/blob/master/data/script.bash) script.

You can directly download the preprocessed datasets from [here](https://drive.google.com/file/d/1a0W8ivmMbHKCn-azBdZS_HxEnGT1CY3f/view?usp=sharing).


## List of dataset to train universal sentence encoder

### [Stanford Natural Language Inference (SNLI) Corpus](http://nlp.stanford.edu/projects/snli/)

<p align="justify">
The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for 
balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference
(NLI), also known as recognizing textual entailment (RTE).
<p align="justify">
  
<p align="justify">
The original data set contains <b>570,152</b> sentence pairs, each labeled with one of the following relationships: entailment, 
contradiction, neutral and –, where – indicates a lack of consensus from the human annotators. Discarding the sentence pairs labeled with – and keeping the remaining ones results in <b>549,367</b> pairs for training, <b>9,842</b> pairs for development and <b>9,824</b> pairs for testing.
<p align="justify">
  
### [Multi-Genre NLI Corpus, or MultiNLI](https://repeval2017.github.io/shared/)

<p align="justify">
The task dataset (called the Multi-Genre NLI Corpus, or MultiNLI) consist of <b>393,000</b> training examples drawn from five genres of text, and <b>40,000</b> test and development examples drawn from those same five genres, as well as five more. Data collection for the task dataset is closely modeled as SNLI.
<p align="justify">

### [Quora Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)

<p align="justify">
Quora question dataset consists of over <b>400,000</b> lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair. Quora dataset split from the paper - <a href="https://github.com/zhiguowang/BiMPM">BiMPM: Bilateral Multi-Perspective Matching for Natural Language Sentences</a> can be found <a href="https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view">here</a>.
<p align="justify">

### [CQADupStack](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

<p align="justify">
CQADupStack is a benchmark dataset for community question-answering (cQA) research. More details can be found in the paper - <a href="http://dl.acm.org/citation.cfm?id=2838934">CQADupStack: A Benchmark Data Set for Community Question-Answering Research</a>.
<p align="justify">
