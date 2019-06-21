#coding:utf-8
'''
define encoder-decoder
define encoder-decoder with attention
define transformer

'''

from __future__ import absolute_import,division,print_function,unicode_literals

import tensorflow as tf 
tf.enable_eager_execution()

import numpy as np 
import os
import time


def gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):

    def __init__(self,vocab_size,embedding_dim,enc_units,batch_sz):
        super(Encoder.self).__init__()

        self._vocab_size = vocab_size
        self._embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self._enc_units = enc_units
        self._batch_sz = batch_sz

        self._gru = gru(self._enc_units)

    ##定义前向传播方法
    def call(self,x,hidden):
        ## 将输入转化为embedding
        x = self.embedding(x)

        ## 使用gru的RNN进行前向传播，得到每一步的输出以及state
        output,state = self.gru(x,initial_state = hidden)
        return output,state

    def initialize_hidden_state(self):
        ## 初始状态是 batch size x enc_units的大小
        return tf.zeros((self._batch_sz,self._enc_units))



class BahdanauAttention(tf.keras.Model):

    def __init__(self,units):
        super(BahdanauAttention.self).__init__()

        self._W1 = tf.keras.layers.Dense(units)

        self._W2 = tf.keras.layers.Dense(units)

        self._V = tf.keras.layers.Dense(1)

    def call(self,query,values):

        hidden_with_time_axis = tf.expand_dims(query,1)

        score = self._V(tf.nn.tanh(self._W1(values)+self._W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score,axis=1)

        context_vector = attention_weights*values

        context_vector = tf.reduce_sum(context_vector,axis=1)

        return context_vector,attention_weights



class Decoder(tf.keras.Model):

    def __init__(self,vocab_size,embedding_dim,dec_units,batch_sz):

        super(Decoder.self).__init__()

        self._batch_sz = batch_sz

        self._dec_units = dec_units
        ## encoder decoder的embedding是不共享的
        self._embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)

        self._gru = gru(self.dec_units)

        self._fc = tf.keras.layers.Dense(vocab_size)

        ## Attention
        self._attention = BahdanauAttention(self._dec_units)

    def call(self,x,hidden,enc_output):

        ## encoder的输出是 batch size x length * hidden size
        context_vector,attention_weights = self._attention(hidden,enc_output)

        ## decoder自己的embedding
        x = self._embedding(x)

        ## 将context vector与就之前的output进行串联
        x = tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)

        output,state = self.gru(x)

        output = tf.reshape(output,(-1,output.shape[2]))

        x = self._fc(output)

        return x,state,attention_weights

    def initialize_hidden_state(self):
        return tf.aeros((self._batch_sz,self._dec_units))

