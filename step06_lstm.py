# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Conv1D, LSTM,\
    Dropout, GaussianNoise, BatchNormalization, Flatten, MaxPooling1D, Bidirectional, Input
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.utils import Sequence, multi_gpu_model
from tensorflow.keras import backend
from gensim.models.doc2vec import Doc2Vec

import multiprocessing
import numpy as np
import random,json
import sys,io,re,os
from time import sleep, time
import argparse
from math import ceil

#変更するとモデル再構築必要
DOC_VEC_SIZE = 128  # Doc2vecの出力より
MAXLEN = 10     # vec推定で参照するトゥート(vecor)数
AVE_LEN = 2

#いろいろなパラメータ
epochs = 10
batch_size = 1024
# 同時実行プロセス数
process_count = multiprocessing.cpu_count() - 1


def lstm_model():
    input_vector = Input(shape=(MAXLEN, DOC_VEC_SIZE))
    layers = Bidirectional(LSTM(1024, return_sequences=True))(input_vector)
    layers = GaussianNoise(0.15)(layers)
    layers = Bidirectional(LSTM(512))(layers)
    layers = Dropout(0.3)(layers)
    layers = Dense(DOC_VEC_SIZE)(layers)

    return Model(inputs=[input_vector], outputs=[layers])


def build_tf_ds(batch_size=1):
    def gen():
        for i in range(len(d2v_vecs) - MAXLEN):
            yield (d2v_vecs[i:i + MAXLEN, :], d2v_vecs[i + MAXLEN,:])

    tf_ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
    tf_ds = tf_ds.cache()
    tf_ds = tf_ds.shuffle(256)
    tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
    tf_ds = tf_ds.prefetch(tf.data.experimental.AUTOTUNE)    
    return tf_ds


def on_epoch_end(epoch, logs):
    ### save
    print('----- saving model...')
    model.save("/content/drive/My Drive/colab/lstm_vec.h5")
    sleep(1)
    model.save(f"/content/drive/My Drive/colab/lstm_vec_{epoch}.h5")
    sleep(1)

    ### test
    start_index = random.randrange(0, d2v_vecs.shape[0] - MAXLEN)
    x_pred = d2v_vecs[start_index:start_index + MAXLEN, :]
    for i in range(MAXLEN):
        ret = d2v_model.docvecs.most_similar([x_pred[i,:]])
        id, score = ret[0]
        print(f"in:{score:3f} {toots[id]}")

    print("ans:",toots[tags[start_index + MAXLEN]])
    print("ans:",d2v_vecs[start_index + MAXLEN, :10])

    x_pred = np.reshape(x_pred,(1,x_pred.shape[0],x_pred.shape[1]))
    preds = model.predict_on_batch(x_pred)
    print(f"pred vec ={preds[0][:10]}")
    ret = d2v_model.docvecs.most_similar(np.asarray(preds))
    for id, score in ret:
        print(f"pred:{score:3f} {toots[id]}")


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

    # モデル構築
    if os.path.exists("/content/drive/My Drive/colab/lstm_vec.h5"):
        # loading the model
        print('load model...')
        model = load_model("/content/drive/My Drive/colab/lstm_vec.h5")
    else:
        model = lstm_model()

    model.compile(loss='mean_squared_error', optimizer=Nadam())
    model.summary()
    m = model
    # d2v ベクトル取得
    d2v_model = Doc2Vec.load("/content/drive/My Drive/colab/d2v.model")
    temp_vecs = d2v_model.docvecs.vectors_docs
    d2v_vecs = np.zeros((temp_vecs.shape[0] - AVE_LEN, temp_vecs.shape[1]))
    for i in range(temp_vecs.shape[0] - AVE_LEN):
        d2v_vecs[i,:] = np.mean(temp_vecs[i:i+AVE_LEN,:], axis=0)
    del(temp_vecs)
    # テキスト取得
    tags = [tag.strip() for tag in open(
        "/content/drive/My Drive/colab/ids_merge.txt").readlines()]
    toots = {tag: toot.strip() for tag, toot in zip(tags, open("/content/drive/My Drive/colab/toot_merge_n.txt").readlines())}
    # データセット構築
    dataset = build_tf_ds(batch_size=batch_size)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    ES = EarlyStopping(monitor='loss', min_delta=0.001,
                       patience=5, verbose=0, mode='auto')
    # トレーニング
    m.fit(dataset,
        callbacks=[print_callback,ES],
        epochs=epochs,
        verbose=1,
        steps_per_epoch=(len(d2v_vecs) - MAXLEN)//batch_size,
        # initial_epoch=,
        max_queue_size=process_count,
        workers=2,
        use_multiprocessing=False)
