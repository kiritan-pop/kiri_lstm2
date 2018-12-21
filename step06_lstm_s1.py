# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.callbacks import LambdaCallback,EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Conv1D, LSTM,\
                    Dropout, GaussianNoise, BatchNormalization , Flatten, MaxPooling1D
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

import tensorflow as tf
graph = tf.get_default_graph()

#変更するとモデル再構築必要
VEC_SIZE = 256  # Doc2vecの出力より
MAXLEN = 5     # vec推定で参照するトゥート(vecor)数
AVE_LEN = 5

#いろいろなパラメータ
epochs = 10000
# 同時実行プロセス数
process_count = multiprocessing.cpu_count() - 1

def lstm_model():
    model = Sequential()
    # model.add(Conv1D(filters=128,kernel_size=8,strides=1, padding='same', input_shape=(MAXLEN, VEC_SIZE)))
    # # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(2, padding='same'))
    # model.add(Conv1D(filters=256,kernel_size=8,strides=1, padding='same'))
    # # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(2, padding='same'))
    # model.add(Conv1D(filters=512,kernel_size=8,strides=1, padding='same'))
    # # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(2, padding='same'))
    # model.add(Conv1D(filters=1024,kernel_size=8,strides=1, padding='same'))
    # # model.add(BatchNormalization())
    # model.add(Activation('tanh'))
    # model.add(Flatten())
    # model.add(Dropout(0.5))
    # model.add(Dense(VEC_SIZE))
    # model.add(Activation('tanh'))


    model.add(LSTM(1024, return_sequences=True, input_shape=(MAXLEN, VEC_SIZE)))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(LSTM(256, return_sequences=True))
    # model.add(Flatten())
    model.add(LSTM(512))
    model.add(Dense(VEC_SIZE))

    # model.add(Dense(VEC_SIZE*128, activation="relu", input_shape=(MAXLEN, VEC_SIZE)))
    # model.add(Flatten())
    # model.add(Dropout(0.3))
    # model.add(Dense(VEC_SIZE))

    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2v_model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--toots_path", type=str)
    parser.add_argument("--tags_path", type=str)
    parser.add_argument("--gpu", type=str, default='1')
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    return args

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
            ret = self.d2v_model.docvecs.most_similar([x_pred[i,:]])
            id, score = ret[0]
            print(f"in:{score:3f} {toots[id]}")

        print("ans:",toots[tags[start_index + MAXLEN]])
        print("ans:",self.vecs[start_index + MAXLEN, :10])

        x_pred = np.reshape(x_pred,(1,x_pred.shape[0],x_pred.shape[1]))
        with graph.as_default():
            preds = model.predict_on_batch(x_pred)
        # print("vec:",preds[0])
        # print(preds.shape)
        print(f"pred vec ={preds[0][:10]}")
        # print(type(preds))
        # print(type(preds[0]))
        ret = self.d2v_model.docvecs.most_similar(preds)
        for id, score in ret:
            print(f"out:{score:3f} {toots[id]}")

def on_epoch_end(epoch, logs):
    ### save
    print('----- saving model...')
    model.save(args.model_path)

if __name__ == '__main__':
    #パラメータ取得
    args = get_args()
    #GPU設定
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False,
                                                    visible_device_list=args.gpu
                                                    ))
    session = tf.Session(config=config)
    backend.set_session(session)

    GPUs = len(args.gpu.split(','))

    d2v_model = Doc2Vec.load(args.d2v_model)

    tags = [tmp.strip() for tmp in open(args.tags_path).readlines()]
    toots = {tag:toot.strip() for tag,toot in zip(tags,open(args.toots_path).readlines())}

    if os.path.exists(args.model_path):
        # loading the model
        print('load model...')
        model =  load_model(args.model_path)
    else:
        model = lstm_model()
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    m = model
    if GPUs > 1:
        p_model = multi_gpu_model(model, gpus=GPUs)
        p_model.compile(loss='mean_squared_error', optimizer=RMSprop())
        m = p_model

    generator = DataGenerator(d2v_model=d2v_model, batch_size=args.batch_size)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    ES = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')

    m.fit_generator(generator,
                    callbacks=[print_callback,ES],
                    epochs=epochs,
                    verbose=1,
                    # steps_per_epoch=60,
                    initial_epoch=args.idx,
                    max_queue_size=process_count,
                    workers=2,
                    use_multiprocessing=False)
