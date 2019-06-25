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
from scipy import stats

from seq2seq_model import Encoder,Decoder


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

        content = content.replace('-','')

        if len(content) < 100:
            continue

        data.append([pos,title,content])

    # print(data[:1])

    #对输入输出进行编号
    input_wix =  WordIndex(content for pos,title,content in data)
    target_wix = WordIndex(title for pos,title,content in data)

    ##对输入输出进行转化
    input_tensor = [[input_wix.word2ix[w] for w in content] for pos,title,content in data]
    target_tensor = [[target_wix.word2ix[w] for w in title] for pos,title,content in data]

    ## 计算输入输出的最大长度
    max_length_inp,max_length_tar = mode_length(input_tensor),max_length(target_tensor)
    print('mode length of input data is %d, and  max length of target data is %d.'% (max_length_inp,max_length_tar))

    ## 根据最大长度对输入输出进行padding
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,maxlen=max_length_inp,padding='post',truncating='post')
    target_tensor= tf.keras.preprocessing.sequence.pad_sequences(target_tensor,maxlen=max_length_tar,padding='post',truncating='post')

    return input_tensor,target_tensor,max_length_inp,max_length_tar,input_wix,target_wix


def mean_length(tensor):
    return int(np.mean([len(t) for t in tensor]))

def mode_length(tensor):
    return int(stats.mode([len(t) for t in tensor])[0])

def max_length(tensor):
    return int(np.max([len(t) for t in tensor]))

## 将所有的字进行index，转化为数字
class WordIndex():

    def __init__(self,words):
        self._words = words

        self.word2ix = {}
        self.ix2word = {}

        self.vocab = set()

        self.create_index()

    def create_index(self):

        ## 每一个字和符号
        for word in self._words:
            for c in word:
                self.vocab.add(c)

        self.vocab = sorted(self.vocab)

        self.word2ix['<pad>'] = 0

        for ix,word in enumerate(self.vocab):
            self.word2ix[word] = ix

        for word,ix in self.word2ix.items():
            self.ix2word[ix] = word


def loss_function(real,pref):
    mask = 1-np.equal(real,0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask

    return tf.reduce_mean(loss)


class S2SM:

    def __init__(self):
        ## 加载数据
        input_tensor,target_tensor,mode_length_inp,max_length_tar,input_wix,target_wix = load_data()
        vocab_inp_size = len(input_wix.word2ix)
        vocab_tar_size = len(target_wix.word2ix)
        ## 分为训练数据和测试数据
        self._input_tensor_train, self._input_tensor_val, self._target_tensor_train, self._target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        ## 超参数
        self._units = 512
        self._batch_sz = 64
        self._embdding_dim = 256
        self._buffer_size = len(self._input_tensor_train)
        self._n_batchs =self._buffer_size//self._batch_sz

        ## 数据集
        self._dataset = tf.data.Dataset.from_tensor_slices((self._input_tensor_train, self._target_tensor_train)).shuffle(self._buffer_size)
        self._dataset = dataset.batch(self._batch_sz, drop_remainder=True)

        ## 初始化encoder以及decoder
        self._encoder = Encoder(vocab_inp_size,self._embdding_dim,self._batch_sz)
        self._decoder = Decoder(vocab_tar_size,self._embdding_dim,self._batch_sz)


    def train(self):

        EPOCHS = 10

        for epoch in range(EPOCHS):
            start = time.time()

            hideen = self._encoder.initialize_hidden_state()
            total_loss = 0

            for (batch,(inp,targ)) in enumerate(self._dataset):
                loss = 0

                with tf.GradientTaper as tape:

                    enc_output, enc_hidden = self._encoder()




if __name__ == '__main__':
    load_data()
