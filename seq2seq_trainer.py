#coding:utf-8
'''
loading data and train seq2seq model

'''


from __future__ import absolute_import,division,print_function,unicode_literals

import tensorflow as tf 
tf.compat.v1.enable_eager_execution()

print(tf.__version__)

import numpy as np 
import os
import time
from scipy import stats
import logging
from sklearn.model_selection import train_test_split


from seq2seq_model import Encoder,Decoder

from rouge import Rouge
rouge = Rouge()
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


## 数据的位置
## 数据的格式是 pos===========================title===========================content
def load_data():

    print('loading data from data/train/train.txt')
    data = []
    progress= 0 
    for line in open('data/train/train.txt',encoding='utf-8'):

        progress+=1

        if progress%1000==1:
            print('Loading progress {:} ...'.format(progress))

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
    print('creating input and target vocabularies ...')
    input_wix =  WordIndex(tokenize(content) for pos,title,content in data)
    target_wix = WordIndex(tokenize(title) for pos,title,content in data)

    ##对输入输出进行转化
    print('covert dataset to tensor ...')
    input_tensor = [[input_wix.word2ix[w] for w in tokenize(content)] for pos,title,content in data]
    target_tensor = [[target_wix.word2ix[w] for w in tokenize(title)] for pos,title,content in data]


    ## 计算输入输出的最大长度
    max_length_inp,max_length_tar = mode_length(input_tensor),mean_length(target_tensor)
    print('mode length of input data is %d, and  max length of target data is %d.'% (max_length_inp,max_length_tar))

    ## 根据最大长度对输入输出进行padding
    print('padding input and target tensor ...')
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,maxlen=max_length_inp,padding='post',truncating='post')
    target_tensor= tf.keras.preprocessing.sequence.pad_sequences(target_tensor,maxlen=max_length_tar,padding='post',truncating='post')

    return input_tensor,target_tensor,max_length_inp,max_length_tar,input_wix,target_wix


def tokenize(str):
    tokens = []

    tokens.append('<start>')
    for c in str:
        tokens.append(c)
    tokens.append('<end>')
    return tokens


def mean_length(tensor):
    return int(np.mean([len(t) for t in tensor]))

def mode_length(tensor):
    return int(stats.mode([len(t) for t in tensor])[0]*0.8)

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


def loss_function(real,pred):
    mask = 1-np.equal(real,0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask

    return tf.reduce_mean(loss)


class S2SM:

    def __init__(self):
        ## 加载数据
        input_tensor,target_tensor,self._length_inp,self._length_tar,self._input_wix,self._target_wix = load_data()
        vocab_inp_size = len(self._input_wix.word2ix)
        vocab_tar_size = len(self._target_wix.word2ix)
        ## 分为训练数据和测试数据
        self._input_tensor_train, self._input_tensor_val, self._target_tensor_train, self._target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        ## 超参数
        self._units = 512
        self._batch_sz = 32
        self._embdding_dim = 200
        self._buffer_size = len(self._input_tensor_train)
        self._n_batchs =self._buffer_size//self._batch_sz

        ## 数据集
        self._dataset = tf.data.Dataset.from_tensor_slices((self._input_tensor_train, self._target_tensor_train)).shuffle(self._buffer_size)
        self._dataset = self._dataset.batch(self._batch_sz, drop_remainder=True)

        ## 初始化encoder以及decoder
        self._encoder = Encoder(vocab_inp_size,self._embdding_dim,self._units,self._batch_sz)
        self._decoder = Decoder(vocab_tar_size,self._embdding_dim,self._units,self._batch_sz)

        ## optimizer
        self._optimizer = tf.keras.optimizers.Adam()

        ## 模型的保存位置
        self._checkpoint_dir = './tranning_checkpoints'
        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
        self._checkpoint = tf.train.Checkpoint(optimizer=self._optimizer,encoder=self._encoder,decoder=self._decoder)


    def reload_latest_checkpoints(self):
        print('reload latest Checkpoint.....')
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_dir))

        # pass

    def train_step(self,inp,targ,enc_hidden):

        loss = 0

        with tf.GradientTape() as tape:

            enc_output, enc_hidden = self._encoder(inp,enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self._target_wix.word2ix['<start>']]*self._batch_sz,1)

            # print('===dec input shape {}'.format(dec_input.shape))

            # print('===dec hidden shape {}'.format(dec_hidden.shape))
            # print('===enc output shape {}'.format(enc_output.shape))

            for t in range(1,targ.shape[1]):

                predictions,dec_hidden,_ = self._decoder(dec_input,dec_hidden,enc_output)

                loss += loss_function(targ[:,t],predictions)

                ## 时间t的标准结果作为t+1的x
                dec_input = tf.expand_dims(targ[:,t],1)

        batch_loss = (loss/int(targ.shape[1]))

        variables = self._encoder.trainable_variables + self._decoder.trainable_variables

        gradients =  tape.gradient(loss,variables)

        self._optimizer.apply_gradients(zip(gradients,variables))

        return batch_loss

    def train(self):

        EPOCHS = 1000

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self._encoder.initialize_hidden_state()

            total_loss = 0

            for (batch,(inp,targ)) in enumerate(self._dataset.take(self._n_batchs)):

                batch_loss  = self.train_step(inp,targ,enc_hidden)

                total_loss+=batch_loss

                if (batch+1)%100==1 or batch==self._n_batchs:

                    print('Epoch {} Batch {}/{} Loss {:.4f}'.format(epoch+1,batch,self._n_batchs,batch_loss.numpy()))


            if (epoch+1)%10==0:

                self._checkpoint.save(file_prefix = self._checkpoint_prefix)

            print('Epoch {} Loss {:.4f}, Average Rouge F1 Score {:.4f}'.format(epoch+1,total_loss/self._n_batchs,self.validate()))

            print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))


    def validate(self):

        all_fs = []
        for ix in np.random.choice(len(self._input_tensor_val),self._batch_sz*10):
            # print(ix)
            _input,_tar = self._input_tensor_val[ix],self._target_tensor_val[ix]

            # print(_input)
            # print(len(_tar))

            ref = []
            ref_ids = []
            for _id in _tar:

                word = self._target_wix.ix2word[_id]
                if word == '<end>':
                    break

                if word == '<start>':
                    continue

                ref.append(word)
                ref_ids.append(_id)

            # print('REF',ref)

            if' '.join(ref).strip()=='':
                continue

            ref = ' '.join([str(i) for i in ref_ids])

            result = []
            hidden = [tf.zeros((1,self._units))]

            _inputs = tf.convert_to_tensor([_input])

            enc_out,enc_hidden = self._encoder(_inputs,hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self._target_wix.word2ix['<start>']],0)


            for t in range(self._length_tar):

                predictions,dec_hidden,attention_weights = self._decoder(dec_input,dec_hidden,enc_out)

                predicted_id = tf.argmax(predictions[0]).numpy()

                predicted_word = self._target_wix.ix2word[predicted_id]
                if predicted_word=='<end>':
                    break

                result.append(predicted_id)

            res = ' '.join([str(i) for i in result])
            # print('RES',res)
            if res.strip()=='':
                continue

            f = rouge.get_scores(res,ref)
            f = f[0]['rouge-l']['f']

            all_fs.append(f)

        return np.mean(all_fs)


    def evaluate(self,sentence):

        attention_plot = np.zeros((self._length_tar,self._length_inp))

        sentence = tokenize(sentence)

        inputs = [self._input_wix.word2ix[s] for s in sentence]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=self._length_tar,padding='post',truncating='post')
        inputs = tf.convert_to_tensor(inputs)

        result= ''

        hidden = [tf.zeros((1,self._units))]

        enc_out,enc_hidden = self._encoder(inputs,hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([self._target_wix.word2ix['<start>']],0)

        for t in range(self._length_tar):

            predictions,dec_hidden,attention_weights = self._decoder(dec_input,dec_hidden,enc_out)

            attention_weights = tf.reshape(attention_weights,(-1,))

            # attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            predicted_word = self._target_wix.ix2word[predicted_id]
            if predicted_word=='<end>':

                return result,sentence,attention_plot

            result+=predicted_word

            

            dec_input = tf.expand_dims([predicted_id],0)

        return result,sentence,attention_plot



def test_eval():

     progress = 0
     for line in open('data/train/train.txt',encoding='utf-8'):

        splits = line.split('===========================')

        if len(splits)!=3:
            continue

        progress+=1

        if progress<5000:
            continue

        pos,title,content = splits

        if len(content)<100:
            continue

        print('-------------------')
        content = content.replace('-','')
        print(content)

        print(title)

        s2s_model = S2SM()

        s2s_model.reload_latest_checkpoints()

        result,sentence,attention_plot = s2s_model.evaluate(content)

        print('RESULT:',result)
        print('GOLDEN:',title)


        break





if __name__ == '__main__':
    # load_data()

    # s2s_model = S2SM()
    # s2s_model.reload_latest_checkpoints()
    # s2s_model.train()

    test_eval()


