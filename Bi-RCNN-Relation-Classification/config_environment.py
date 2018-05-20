stanford_corenlp_root_path = r"C:/stanford-corenlp-full-2017-06-09/stanford-corenlp-full-2017-06-09/"


word_wec_path = "C:/Users/Administrator/Desktop/wordwec/"
# copus__dimension__vocabulary size__author__architecture__training algorithm__context window
# ----------------------------------  Word2Vec  ------------------------------------------------------
# Google News___300d___3M___Google___word2vec___negative sampling___BoW - ~5
word_vec_file1 = word_wec_path + "deps.words/deps.words"

# -----------------------------------  GloVe  --------------------------------------------------------
# Twitter___50d___1.2M___Glove___GloVe___GloVe___AdaGrad
word_vec_file2 = word_wec_path + "glove.6B/glove.6B.50d.txt"
# Twitter___100d___1.2M___Glove___GloVe___GloVe___AdaGrad
word_vec_file3 = word_wec_path + "glove.6B/glove.6B.100d.txt"
# Twitter___200d___1.2M___Glove___GloVe___GloVe___AdaGrad
word_vec_file4 = word_wec_path + "glove.6B/glove.6B.200d.txt"

# ------------------------------------  GloVe  -------------------------------------------------------
# Wikipedia+Gigaword5___50d___400,000___Glove___GloVe___AdaGrad___10+10
word_vec_file5 = word_wec_path + "glove.twitter.27B/glove.twitter.27B.50d.txt"
# Wikipedia+Gigaword5___100d___400,000___Glove___GloVe___AdaGrad___10+10
word_vec_file6 = word_wec_path + "glove.twitter.27B/glove.twitter.27B.100d.txt"
# Wikipedia+Gigaword5___200d___400,000___Glove___GloVe___AdaGrad___10+10
word_vec_file7 = word_wec_path + "glove.twitter.27B/glove.twitter.27B.200d.txt"

# 存成字典，方便索引
word_vec_file = \
    [word_vec_file1, word_vec_file2, word_vec_file3, word_vec_file4, word_vec_file5, word_vec_file6, word_vec_file7]
# 生成数据时候的文件名
word_vec_file_state = ["deps", "glove.6B.50d", "glove.6B.100d", "glove.6B.200d",
                       "glove.twitter.27B.50d", "glove.twitter.27B.100d", "glove.twitter.27B.200d"]


# list中项的顺序不要动，会翻车的
cat_names = \
    ['Cause-Effect(e1,e2)', 'Component-Whole(e1,e2)', 'Content-Container(e1,e2)', 'Entity-Destination(e1,e2)',
     'Entity-Origin(e1,e2)', 'Instrument-Agency(e1,e2)', 'Member-Collection(e1,e2)', 'Message-Topic(e1,e2)',
     'Product-Producer(e1,e2)', 'Other', 'Cause-Effect(e2,e1)', 'Component-Whole(e2,e1)', 'Content-Container(e2,e1)',
     'Entity-Destination(e2,e1)', 'Entity-Origin(e2,e1)', 'Instrument-Agency(e2,e1)', 'Member-Collection(e2,e1)',
     'Message-Topic(e2,e1)', 'Product-Producer(e2,e1)']


# label标注为 0-18 在后面转化成稀疏向量的时候，不会发生索引越界
def create_cat_map(cat_names):
    cat_map = {}
    label = 0
    for cat in cat_names:
        cat_map[cat_names[label]] = label
        label += 1
    return cat_map





