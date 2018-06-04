#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: stanford.py 
@desc:
@time: 2018/06/04 
"""

import os
class StanfordCoreNLP:
    def __init__(self, jar_path):
        self.root = jar_path
        self.tmp_src_path = "tmpsrc"  # 临时文件目录
        self.jar_list = ["ejml-0.23.jar", "javax.json.jar", "jollyday.jar", "joda-time.jar", "protobuf.jar",
                         "slf4j-api.jar", "slf4j.simple.jar", "stanford-corenlp-3.8.0.jar", "xom.jar"]
        self.jar_path = ""
        self.build_jars()

    def build_jars(self):  # 根据root路径构建所有的jar路径包
        for jar in self.jar_list:
            self.jar_path += self.root + jar + ";"

    def save_file(self, path, sent):  # 创建临时文件存储路径
        fp = open(path, "wb")
        sent_ = bytes(sent, encoding="utf-8")
        fp.write(sent_)
        fp.close()

    def del_file(self, path):  # 删除临时文件
        os.remove(path)

class StanfordParser(StanfordCoreNLP):
    def __init__(self, model_path, jar_path, opt_type, opt_options):
        StanfordCoreNLP.__init__(self, jar_path)
        self.model_path = model_path
        self.classfier = "edu.stanford.nlp.parser.lexparser.LexicalizedParser"
        self.opt_type = opt_type
        self.opt_options = opt_options
        self.__build_cmd()

    # 构建命令行
    def __build_cmd(self):
        self.cmd_line = 'java -mx1g -cp "' + self.jar_path + '" ' + self.classfier + \
                        ' -retainTmpSubcategories -originalDependencies -outputFormat "' + self.opt_type + \
                        '" -outputFormatOptions "' + self.opt_options + '" ' + \
                        self.model_path + ' '

    def parse(self, sent):
        '''
        解析句子
        :param sent:
        :return:
        '''
        self.save_file(self.tmp_src_path, sent)
        tag_txt = os.popen(self.cmd_line + self.tmp_src_path, "r").read()  # 输出到变量中
        self.del_file(self.tmp_src_path)
        return tag_txt

        # 输出到文件

    def tag_file(self, sent, out_path):
        self.save_file(self.tmp_src_path, sent)
        os.system(self.cmd_line + self.tmp_src_path + ' > ' + out_path)
        self.del_file(self.tmp_src_path)