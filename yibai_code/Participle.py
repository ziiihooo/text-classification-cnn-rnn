from msilib.schema import Directory
import os
import os.path
import jieba
import json

from numpy import append

split_word = [' ',"\u3000","\n"]

class Participle:
    # 读取停用词
    def load_stopwords(self):
        with open("./yibai_code/stop_words.txt") as F:
            stopwords=F.readlines()
            F.close()
        return [word.strip() for word in stopwords]

    #  读取文件内容核文件路径，存储为字典
    def load_files(self):
        fileContents = {}
        for root, dirs, files in os.walk(
            r"./data/cnews/"
        ):
            filePaths = []
            fileContent_eachdir = {}
            for index,name in enumerate(files):
                filePaths.append(os.path.join(root, name))
                f = open(filePaths[index], encoding= 'utf-8')
                file_subjects = {}
                i = 0
                while True:
                    fileContent = f.readline()
                    if not fileContent:
                        break
                    ret = fileContent.split("\t")
                    if(file_subjects.get(ret[0]) == None):
                        i=0
                        file_subjects[ret[0]] = []
                    file_subjects[ret[0]].append({i:ret[1]})  
                    i = i+1                  
                f.close()
                fileContent_eachdir[name] = file_subjects
            if fileContent_eachdir:
                fileContents = fileContent_eachdir
        return fileContents

    def write2json(self):
        result2file = json.dumps(self.seg_result,ensure_ascii=False)
        f = open('seg_result_json.json',encoding="utf-8", mode='w')
        f.write(result2file)
        f.close()
        return

    def read_json2directory(self):
        f = open('seg_result_json.json',encoding="utf-8", mode='r')
        content = f.read()
        self.seg_result = json.loads(content)
        f.close()
        return

    # 分词函数
    def divide_delete_words(self,file_contents,stop_word):
        result = {}
        # 遍历文件夹
        for file_eachdir in file_contents:
            file_seg_list = {}
            # 遍历文件
            for file_name in file_contents[file_eachdir]:
                file_seg_list[file_name] = {}
                for index,content_name in enumerate(file_contents[file_eachdir][file_name]):
                    none_stop_list = []
                    # print(file_contents[file_eachdir][file_name])
                    # 分词函数
                    seg_list = jieba.cut(file_contents[file_eachdir][file_name][index][index])
                    #筛选停用词和特殊换行空格词
                    for item in seg_list:
                        if(item not in stop_word and item not in split_word):
                           none_stop_list.append(item)
                    # 添加该文件所有分词结果
                    file_seg_list[file_name][index] = none_stop_list
            # 添加该文件夹所有文件
            result[file_eachdir] = file_seg_list
        return result

    def __init__(self,is_read_file = False) -> Directory:
        if not is_read_file:
            stop_word = self.load_stopwords()
            file_content = self.load_files()
            self.seg_result = self.divide_delete_words(file_content,stop_word) 
            self.write2json()
        elif is_read_file:
            self.read_json2directory()
        return 