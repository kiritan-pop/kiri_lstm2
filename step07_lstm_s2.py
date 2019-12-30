# -*- coding: utf-8 -*-
from keras.models import Sequential,load_model,Model
from keras.callbacks import LambdaCallback,EarlyStopping
from keras.layers import Dense, Activation, CuDNNLSTM, LSTM, Dropout,\
     GaussianNoise, BatchNormalization, Embedding, Flatten, Input, Concatenate, Reshape
from keras.optimizers import RMSprop
from keras.utils import Sequence, multi_gpu_model
from keras import backend
# from tensorflow.python.keras.preprocessing.text import Tokenizer

import multiprocessing
import numpy as np
import random,json
import sys,io,re,os
from time import sleep
import argparse
import math
from gensim.models.doc2vec import Doc2Vec

import tensorflow as tf
graph = tf.get_default_graph()

#変更するとモデル再構築必要
VEC_SIZE = 256  # 文字ベクトル次元／トゥートベクトル次元
MAXLEN = 5      # timestep
MU = "🧪"       # 無
END = "🦷"      # 終わりマーク

#いろいろなパラメータ
epochs = 10000
# 同時実行プロセス数
process_count = multiprocessing.cpu_count() - 1

def lstm_model():
    num_chars = len(wl_chars)

    input_chars = Input(shape=(MAXLEN,))
    layers = Embedding(input_dim=num_chars+2,
                        output_dim=VEC_SIZE,
                        input_length=MAXLEN)(input_chars)

    input_vector = Input(shape=(VEC_SIZE,))
    vector = Reshape(target_shape=(1, VEC_SIZE))(input_vector)

    layers = Concatenate(axis=1)([vector, layers])
    # layers = BatchNormalization()(layers)
    layers = GaussianNoise(0.1)(layers)
    layers = LSTM(1024, return_sequences=True)(layers)
    layers = LSTM(512)(layers)
    layers = Dropout(0.3)(layers)
    layers = Dense(num_chars+2, activation='softmax')(layers)

    return Model(inputs=[input_vector, input_chars], outputs=[layers])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--input", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--d2v_path", type=str)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    return args

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class DataGenerator(Sequence):
    def __init__(self, toots_path, d2v_model, batch_size=1, step=1):
        # コンストラクタ
        self.idx_char = {i:c for i,c in enumerate(wl_chars)}
        self.num_chars = len(self.idx_char)
        self.idx_char[self.num_chars] = MU
        self.idx_char[self.num_chars+1] = END
        self.char_idx = {c:i for i,c in enumerate(wl_chars)}
        self.char_idx[MU] = self.num_chars
        self.char_idx[END] = self.num_chars + 1
        self.vecs = d2v_model.docvecs.vectors_docs
        self.toots = list([tmp.strip() for tmp in open(toots_path).readlines()])
        self.x_vecs_id = []
        self.x_idxs = []
        self.y_next_idx = []
        for id,toot in enumerate( self.toots):
            tmp_chars = MU * MAXLEN
            for next_char in toot+END:
                tmp_idxs = []
                try:
                    for char in tmp_chars:
                        tmp_idxs.append(self.char_idx[char])
                    tmpidx = self.char_idx[next_char]
                except Exception:
                    pass
                else:
                    self.x_vecs_id.append(id)
                    self.x_idxs.append(tmp_idxs)
                    self.y_next_idx.append(tmpidx)

                tmp_chars = tmp_chars[1:] + next_char

        self.batch_size = batch_size
        self.step = step

    def __getitem__(self, idx):
        # データの取得実装
        vecs = [self.vecs[i] for i in  self.x_vecs_id[self.batch_size*idx:self.batch_size*(idx+1)]]
        tmp_mat = [] 
        for j in self.y_next_idx[self.batch_size*idx:self.batch_size*(idx+1)]:
            mat = np.zeros((self.num_chars+2))
            mat[j] = 1
            tmp_mat.append(mat)

        # idx = idx % self.__len__()
        return [np.asarray(vecs),\
                np.asarray(self.x_idxs[self.batch_size*idx:self.batch_size*(idx+1)])],\
                 np.asarray(tmp_mat)

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！
        deta_len = len(self.x_vecs_id)
        sample_per_epoch = math.ceil(deta_len/self.batch_size)
        return sample_per_epoch

    def on_epoch_end(self):
    # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch')

        # start_index = random.randrange(0, len(self.x_idxs))
        starts = random.sample(range(0, len(self.vecs)),5)

        for start_index in starts:
            vec = self.vecs[start_index]
            toot = self.toots[start_index]

            for diversity in [0.1, 0.25, 0.4]:
                print()
                print('----- diversity:', diversity)

                generated = ''
                idxs = [self.char_idx[MU] for _ in range(MAXLEN)]
                print('-----  toot on input vec: "' + toot + '"')
                sys.stdout.write(generated)

                for _ in range(50):
                    with graph.as_default():
                        preds = model.predict_on_batch([ np.asarray([vec]),  np.asarray([idxs]) ])
                        next_index = sample(preds[0], diversity)
                        idxs = idxs[1:]
                        idxs.append(next_index)
                        next_char = self.idx_char[next_index]
                        generated += next_char
                        sys.stdout.write(next_char)
                        sys.stdout.flush()
                        if next_char == END:
                            break
                print()

def on_epoch_end(epoch, logs):
    ### save
    sleep(5)
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
    wl_chars = list(open('./out/wl.txt').read())

    if os.path.exists(args.model_path):
        # loading the model
        print('load model...')
        model =  load_model(args.model_path)
    else:
        model = lstm_model()
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop()) #mean_squared_error
    m = model
    if GPUs > 1:
        p_model = multi_gpu_model(model, gpus=GPUs)
        p_model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
        m = p_model

    d2v_model = Doc2Vec.load(args.d2v_path)
    generator = DataGenerator(toots_path=args.input, d2v_model=d2v_model, batch_size=args.batch_size, step=args.step)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    ES = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=0, mode='auto')

    if args.mode == 'train':
        m.fit_generator(generator,
                        callbacks=[print_callback,ES],
                        epochs=epochs,
                        verbose=1,
                        # steps_per_epoch=60,
                        initial_epoch=args.idx,
                        max_queue_size=process_count,
                        workers=2,
                        use_multiprocessing=False)
    else:
        generator.on_epoch_end()