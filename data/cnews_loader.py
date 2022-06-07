# coding: utf-8

import random
import sys
print(sys.path)

from collections import Counter
from imp import reload

from yibai_code.Participle import *


import numpy as np
import tensorflow.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

# 不用
def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word

# 不用
def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

# 不用
def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

# 改为调用一白函数， 将 字典 根据入参（test.txt, ..） 拆分为content [] 和 labels []
def read_file(filename):
    """读取文件数据"""

    json_dic = Participle(is_read_file=True).seg_result
    contents, labels = [], []

    texts = []

    # 字典转数组 （每个文章list 最后一个为label）
    for cate in json_dic[filename]:
        for text_id in json_dic[filename][cate]:
            text_and_label = list(json_dic[filename][cate][text_id])
            text_and_label.append(cate)
            texts.append(text_and_label)

    # 数组打乱
    random.shuffle(texts)

    for text in texts:
        contents.append(text[:-1])
        labels.append(text[-1])

    return contents, labels

# train_dir 入参为（test.txt, ..）， vocab_dir 为字典地址
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    # 提取words
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


#可以不变， vocab_dir 为字典地址
def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    # 生成dic类型字典
    return words, word_to_id


# 再看看怎么用
def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐', '社会', '股票']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


#可以用
def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


# filename 为（test.txt, ..）， 转换 文本列表 为 向量 ， padding 为固定长度 max_length
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
