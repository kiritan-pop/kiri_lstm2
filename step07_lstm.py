# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import LambdaCallback,EarlyStopping
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout,\
    GaussianNoise, BatchNormalization, Embedding, Flatten, Input, Concatenate, Reshape, Bidirectional
from tensorflow.keras.optimizers import RMSprop, Nadam, Adam
from tensorflow.keras.utils import Sequence, multi_gpu_model
from tensorflow.keras import backend

import multiprocessing
import numpy as np
import random,json
import sys,io,re,os
from time import sleep, time
import argparse
import math
from gensim.models.doc2vec import Doc2Vec

#変更するとモデル再構築必要
DOC_VEC_SIZE = 32 # Doc2vecの出力より
VEC_SIZE = 64  # 文字ベクトル次元
MAXLEN = 5      # timestep
MU = "🧪"       # 無
END = "🦷"      # 終わりマーク

#いろいろなパラメータ
epochs = 30
batch_size = 2**12
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


def build_tf_ds(batch_size=1024):
    def gen():
        x1, x2, y = [], [], []
        for id, toot in enumerate(toots):
            tmp_chars = MU * MAXLEN
            for i, next_char in enumerate(toot+END):
                tmp_idxs = []
                for char in tmp_chars:
                    tmp_idxs.append(char_idx[char])
                nextidx = char_idx[next_char]
                # yield ((d2v_vecs[id], tmp_idxs), tf.one_hot(nextidx, num_chars+2))
                x1.append(d2v_vecs[id])
                x2.append(tmp_idxs)
                y.append(nextidx)
                tmp_chars = tmp_chars[1:] + next_char
        return np.asarray(x1), np.asarray(x2), y
    # tf_ds = tf.data.Dataset.from_generator(gen, ((tf.float32, tf.uint32), tf.uint8))
    tf_dsX1, tf_dsX2, tf_dsY = gen()
    print(len(tf_dsX1))
    tf_dsX1 = tf.data.Dataset.from_tensor_slices(tf.cast(tf_dsX1, tf.float32))
    tf_dsX2 = tf.data.Dataset.from_tensor_slices(tf.cast(tf_dsX2, tf.uint16))
    tf_dsX = tf.data.Dataset.zip((tf_dsX1, tf_dsX2))
    tf_dsY = tf.data.Dataset.from_tensor_slices(
        tf.one_hot(tf_dsY, num_chars+2))
    tf_ds = tf.data.Dataset.zip((tf_dsX, tf_dsY))
    # tf_ds = tf_ds.cache()
    # tf_ds = tf_ds.cache(".cache")
    tf_ds = tf_ds.shuffle(256)
    tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
    tf_ds = tf_ds.shuffle(64)
    tf_ds = tf_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return tf_ds


def on_epoch_end(epoch, logs):
    ### save
    print('----- saving model...')
    model.save(f'/content/drive/My Drive/colab/lstm_set.h5')
    sleep(3)
    model.save(f'/content/drive/My Drive/colab/lstm_set_{epoch}.h5')

    ### test
    sleep(3)
    print()
    print('----- Generating text after Epoch')
    starts = random.sample(range(0, len(d2v_vecs)),5)
    for start_index in starts:
        vec = d2v_vecs[start_index]
        toot = toots[start_index]
        for diversity in [0.1, 0.25, 0.4]:
            print()
            print('----- diversity:', diversity)
            generated = ''
            idxs = [char_idx[MU] for _ in range(MAXLEN)]
            print('-----  toot on input vec: "' + toot + '"')
            sys.stdout.write(generated)
            for _ in range(50):
                preds = model.predict_on_batch([ np.asarray([vec]),  np.asarray([idxs]) ])
                next_index = sample(preds[0], diversity)
                idxs = idxs[1:]
                idxs.append(next_index)
                next_char = idx_char[next_index]
                generated += next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
                if next_char == END:
                    break
            print()


if __name__ == '__main__':
    gpu_id = 0
    print(tf.__version__)
    try:
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
    except Exception as e:
        print(e)

    # 使用する文字種
    wl_chars = list(open('/content/drive/My Drive/colab/wl.txt').read())
    idx_char = {i:c for i,c in enumerate(wl_chars)}
    num_chars = len(idx_char)
    idx_char[num_chars] = MU
    idx_char[num_chars+1] = END
    char_idx = {c:i for i,c in enumerate(wl_chars)}
    char_idx[MU] = num_chars
    char_idx[END] = num_chars + 1
    # モデル構築
    if os.path.exists('/content/drive/My Drive/colab/lstm_set.h5'):
        # loading the model
        print('load model...')
        model = load_model('/content/drive/My Drive/colab/lstm_set.h5')
    else:
        model = lstm_model()
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam())  # mean_squared_error
    m = model
    # テキスト文章取得
    toots = list([tmp.strip() for tmp in open("/content/drive/My Drive/colab/toot_merge_n.txt").readlines()])
    # d2vモデル ベクトル取得
    d2v_vecs = Doc2Vec.load("/content/drive/My Drive/colab/d2v.model").docvecs.vectors_docs
    # データセット構築
    dataset = build_tf_ds(batch_size=batch_size)
    # コールバック設定
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    ES = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
    # トレーニング
    m.fit(dataset,
        callbacks=[print_callback,ES],
        epochs=epochs,
        verbose=1,
        # steps_per_epoch=10,
        # initial_epoch=0,
        # max_queue_size=process_count,
        # workers=2,
        # use_multiprocessing=False
        )
