# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import LambdaCallback,EarlyStopping
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout,\
    GaussianNoise, BatchNormalization, Embedding, Flatten, Input, Concatenate, Reshape, Bidirectional
from tensorflow.keras.optimizers import RMSprop,Nadam
from tensorflow.keras.utils import Sequence, multi_gpu_model
from tensorflow.keras import backend

import multiprocessing
import numpy as np
import random,json
import sys,io,re,os
from time import sleep
import argparse
import math
from gensim.models.doc2vec import Doc2Vec

#変更するとモデル再構築必要
DOC_VEC_SIZE = 128 # Doc2vecの出力より
VEC_SIZE = 256  # 文字ベクトル次元／トゥートベクトル次元
MAXLEN = 5      # timestep
MU = "🧪"       # 無
END = "🦷"      # 終わりマーク

#いろいろなパラメータ
epochs = 30
# 同時実行プロセス数
process_count = multiprocessing.cpu_count() - 1

def lstm_model():
    num_chars = len(wl_chars)

    input_chars = Input(shape=(MAXLEN,))
    layers = Embedding(input_dim=num_chars+2,
                        output_dim=VEC_SIZE,
                        input_length=MAXLEN)(input_chars)

    input_vector = Input(shape=(DOC_VEC_SIZE,))
    docvec = Dense(VEC_SIZE)(input_vector)
    docvec = Reshape(target_shape=(1, VEC_SIZE))(docvec)
    layers = Concatenate(axis=1)([docvec, layers])
    layers = Bidirectional(LSTM(1024, return_sequences=True))(layers)
    layers = GaussianNoise(0.15)(layers)
    layers = Bidirectional(LSTM(512))(layers)
    layers = Dropout(0.3)(layers)
    layers = Dense(num_chars+2, activation='softmax')(layers)

    return Model(inputs=[input_vector, input_chars], outputs=[layers])

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
    model.save(f'/content/drive/My Drive/colab/lstm_set.h5')
    sleep(5)
    model.save(f'/content/drive/My Drive/colab/lstm_set_{epoch}.h5')

if __name__ == '__main__':
    gpu_id = 0
    print(tf.__version__)
    if tf.__version__ >= "2.1.0":
        if 'COLAB_TPU_ADDR' in os.environ:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
            tf.config.experimental_connect_to_cluster(resolver)
            # This is the TPU initialization code that has to be at the beginning.
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
        else:
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(
                physical_devices[gpu_id], True)
    elif tf.__version__ >= "2.0.0":
        #TF2.0
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(
            physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(
            physical_devices[gpu_id], True)
    else:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=str(gpu_id),  # specify GPU number
                allow_growth=True
            )
        )
        set_session(tf.Session(config=config))

    wl_chars = list(open('/content/drive/My Drive/colab/wl.txt').read())

    if os.path.exists('/content/drive/My Drive/colab/lstm_set.h5'):
        # loading the model
        print('load model...')
        model = load_model('/content/drive/My Drive/colab/lstm_set.h5')
    else:
        model = lstm_model()
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Nadam())  # mean_squared_error
    m = model

    d2v_model = Doc2Vec.load("/content/drive/My Drive/colab/d2v.model")
    generator = DataGenerator(
        toots_path="/content/drive/My Drive/colab/toot_merge_n.txt", d2v_model=d2v_model, batch_size=4096, step=1)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    ES = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=0, mode='auto')

    m.fit(generator,
                    callbacks=[print_callback,ES],
                    epochs=30,
                    verbose=1,
                    # steps_per_epoch=60,
                    initial_epoch=0,
                    max_queue_size=process_count,
                    workers=2,
                    use_multiprocessing=False)
