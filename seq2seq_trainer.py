#coding:utf-8
'''
loading data and train seq2seq model

'''


from __future__ import absolute_import,division,print_function,unicode_literals

import tensorflow as tf 
tf.enable_eager_execution()

import numpy as np 
import os
import time

## 数据的位置
## 数据的格式是 pos===========================title===========================content
def load_data():

    data = []
    for line in open('data/train/train.txt',encoding='utf-8'):
        # print(line)
        splits = line.split('===========================')

        if len(splits)!=3:
            continue


        pos,title,content = splits

        # poses.append(pos)
        # titles.append(title)
        # contents.append(content)

        data.append(pos,title,content)

    #对输入输出进行编号
    input_wix =  WordIndex([content for pos,title,content in data])
    target_wix = WordIndex([title for pos,title,content in data])

    ##对输入输出进行转化
    input_tensor = [[input_wix[w] for w in title for pos,title,content in data]]
    target_tensor = [target_wix[w] for w in content for pos,title,content in data]

    ## 计算输入输出的最大长度
    max_length_inp,max_length_tar = max_length(input_tensor),max_length(target_tensor)
    print('max length of input data is %d, and  max length of target data is %d.'% (max_length_inp,max_length_tar))

    ## 根据最大长度对输入输出进行padding
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,maxlen=max_length_inp,padding='post')
    target_tensor= tf.keraas.preprocessing.sequence.pad_sequences(target_tensor,maxlen=max_length_tar,padding='post')


    return input_tensor,output_tensor,max_length_inp,max_length_tar,input_wix,target_wix


def max_length(tensor):
    return max(len(t) for t in tensor)


## 将所有的字进行index，转化为数字
class WordIndex:

    def __init__(self,words):
        self._words = words

        self.word2ix = {}
        self.ix2word = {}

        self.vocab = set()

        self.create_index()

    def create_index(self):

        ## 每一个字和符号
        for word in self._words:
            self.vocab.add(word)

        self.vocab = sorted(self.vocab)

        self.word2ix['<pad>'] == 0

        for ix,word in enumerate(self.vocab):
            self.word2ix[word] = ix

        for word,ix in self.word2ix.items():
            self.ix2word[ix] = word





if __name__ == '__main__':
    load_data()
