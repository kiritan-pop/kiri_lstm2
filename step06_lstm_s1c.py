# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Conv1D, LSTM,\
    Dropout, GaussianNoise, BatchNormalization, Flatten, MaxPooling1D, Bidirectional, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import Sequence, multi_gpu_model
from tensorflow.keras import backend
from gensim.models.doc2vec import Doc2Vec

import multiprocessing
import numpy as np
import random,json
import sys,io,re,os
from time import sleep
import argparse
from math import ceil

#変更するとモデル再構築必要
DOC_VEC_SIZE = 128  # Doc2vecの出力より
MAXLEN = 10     # vec推定で参照するトゥート(vecor)数
AVE_LEN = 2

#いろいろなパラメータ
epochs = 10
batch_size = 8192
# 同時実行プロセス数
process_count = multiprocessing.cpu_count() - 1

def lstm_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(1024, return_sequences=True
                                 ),input_shape=(MAXLEN, DOC_VEC_SIZE)))
    model.add(GaussianNoise(0.15))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dropout(0.3))
    model.add(Dense(DOC_VEC_SIZE))

    return model

def build_tf_ds(d2v_model, batch_size=1):
    temp_vecs = d2v_model.docvecs.vectors_docs
    tf_ds = tf.data.Dataset.from_tensor_slices(tf.range(len(temp_vecs) - MAXLEN))
    tf_ds = tf_ds.map(lambda x: (temp_vecs[x.value_index:x.value_index + MAXLEN, :], temp_vecs[x.value_index + MAXLEN,:]) )
    tf_ds = tf_ds.batch(batch_size)
    tf_ds = tf_ds.shuffle(batch_size)

class DataGenerator(Sequence):
    def __init__(self, d2v_model, batch_size=1, step=1):
        # コンストラクタ
        self.d2v_model = d2v_model
        self.batch_size = batch_size
        temp_vecs = d2v_model.docvecs.vectors_docs
        self.vecs = np.zeros((temp_vecs.shape[0] - AVE_LEN, temp_vecs.shape[1]))
        for i in range(temp_vecs.shape[0] - AVE_LEN):
            self.vecs[i,:] = np.mean(temp_vecs[i:i+AVE_LEN,:], axis=0)


    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！
        deta_len = self.vecs.shape[0] - MAXLEN
        sample_per_epoch = ceil(deta_len/self.batch_size) 
        return sample_per_epoch


    def __getitem__(self, idx):
        # データの取得実装
        x = []
        y = []
        # tmp = range(self.batch_size*idx,min([self.vecs.shape[0]-MAXLEN, self.batch_size*(idx+1)]))
        # print(f"{idx}:{min(tmp)}〜{max(tmp)}")
        # print(self.vecs[self.batch_size*idx,:])
        # print(self.vecs[self.batch_size*idx+MAXLEN,:])
        for i in range(self.batch_size*idx,min([self.vecs.shape[0] - MAXLEN, self.batch_size*(idx+1)])):
            x.append(self.vecs[i:i + MAXLEN, :])
            y.append(self.vecs[i + MAXLEN, :])

        return np.asarray(x), np.asarray(y)

    def on_epoch_end(self):
    # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch')

        start_index = random.randrange(0, self.vecs.shape[0] - MAXLEN)
        # x_pred = []
        # x_pred.append(self.vecs[start_index:start_index + MAXLEN, :])
        x_pred = self.vecs[start_index:start_index + MAXLEN, :]
        # for i in range(start_index,start_index+MAXLEN):
        #     print("in :",toots[tags[i]])
        for i in range(MAXLEN):
            # print(type([x_pred[i, :]]))
            ret = self.d2v_model.docvecs.most_similar([x_pred[i,:]])
            id, score = ret[0]
            print(f"in:{score:3f} {toots[id]}")

        print("ans:",toots[tags[start_index + MAXLEN]])
        print("ans:",self.vecs[start_index + MAXLEN, :10])

        x_pred = np.reshape(x_pred,(1,x_pred.shape[0],x_pred.shape[1]))
        preds = model.predict_on_batch(x_pred)
        # print("vec:",preds[0])
        # print(preds.shape)
        print(f"pred vec ={preds[0][:10]}")
        # print(type(preds))
        # print(type(preds[0]))
        # print(type(preds))
        ret = self.d2v_model.docvecs.most_similar(np.asarray(preds))
        for id, score in ret:
            print(f"out:{score:3f} {toots[id]}")

def on_epoch_end(epoch, logs):
    ### save
    print('----- saving model...')
    model.save("/content/drive/My Drive/colab/lstm_vec.h5")

if __name__ == '__main__':
    gpu_id = 0
    print(tf.__version__)
    if tf.__version__ >= "2.1.0":
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    elif tf.__version__ >= "2.0.0":
        #TF2.0
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    else:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=str(gpu_id),  # specify GPU number
                allow_growth=True
            )
        )
        set_session(tf.Session(config=config))

    d2v_model = Doc2Vec.load("/content/drive/My Drive/colab/d2v.model")

    tags = [tag.strip() for tag in open(
        "/content/drive/My Drive/colab/ids_merge.txt").readlines()]
    toots = {tag: toot.strip() for tag, toot in zip(tags, open("/content/drive/My Drive/colab/toot_merge_n.txt").readlines())}

    generator = DataGenerator(d2v_model=d2v_model, batch_size=batch_size)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    ES = EarlyStopping(monitor='loss', min_delta=0.001,
                       patience=5, verbose=0, mode='auto')

    if os.path.exists("/content/drive/My Drive/colab/lstm_vec.h5"):
        # loading the model
        print('load model...')
        model = load_model("/content/drive/My Drive/colab/lstm_vec.h5")
    else:
        model = lstm_model()

    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    # model.summary()
    m = model

    m.fit(generator,
                    callbacks=[print_callback,ES],
                    epochs=epochs,
                    verbose=1,
                    # steps_per_epoch=10,
                    # initial_epoch=,
                    max_queue_size=process_count,
                    workers=2,
                    use_multiprocessing=True)
