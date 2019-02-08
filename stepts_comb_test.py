# coding: utf-8

from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from gensim.models.doc2vec import Doc2Vec
import MeCab
import numpy as np
import random,json
import sys,io,re,gc
import tensorflow as tf
config = tf.ConfigProto(device_count={"GPU":1},
                        gpu_options=tf.GPUOptions(allow_growth=False, visible_device_list="2"))
session = tf.Session(config=config)
backend.set_session(session)

#いろいろなパラメータ
#変更するとモデル再構築必要
VEC_SIZE = 256  # Doc2vecの出力より
VEC_MAXLEN = 5     # vec推定で参照するトゥート(vecor)数
AVE_LEN = 5        # vec推定で参照するトゥート(vecor)数
TXT_MAXLEN = 5      # 
MU = "🧪"       # 無
END = "🦷"      # 終わりマーク
tagger = MeCab.Tagger('-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd -u ../deep/dic/nicodic.dic')

pat3 = re.compile(r'^\n')
pat4 = re.compile(r'\n')

#辞書読み込み
wl_chars = list(open('wl.txt').read())
idx_char = {i:c for i,c in enumerate(wl_chars)}
num_chars = len(idx_char)
idx_char[num_chars] = MU
idx_char[num_chars+1] = END
char_idx = {c:i for i,c in enumerate(wl_chars)}
char_idx[MU] = num_chars
char_idx[END] = num_chars + 1

d2v_path = 'd2v.model'
lstm_vec_path = 'lstm_vec.h5'
lstm_set_path = 'lstm_set.h5'

d2vmodel = Doc2Vec.load(d2v_path)
lstm_vec_model = load_model(lstm_vec_path)
lstm_set_model = load_model(lstm_set_path)

graph = tf.get_default_graph()


def sample(preds, temperature=1.2):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def lstm_gentxt(toots,num=0,sel_model=None):
    # 入力トゥート（VEC_MAXLEN）をベクトル化。
    input_vec = np.zeros((VEC_MAXLEN + AVE_LEN, VEC_SIZE))
    input_mean_vec = np.zeros((VEC_MAXLEN, VEC_SIZE))
    if len(toots) >= VEC_MAXLEN + AVE_LEN:
        toots_nrm = toots[-(VEC_MAXLEN + AVE_LEN):]
    else:
        toots_nrm = toots + [toots[-1]]*(VEC_MAXLEN + AVE_LEN -len(toots))

    for i,toot in enumerate(toots_nrm):
        wakati = tagger.parse(toot).split(" ")
        input_vec[i] = d2vmodel.infer_vector(wakati)

    for i in range(VEC_MAXLEN):
        input_mean_vec[i] = np.mean(input_vec[i:i+AVE_LEN], axis=0)

    # ベクトル推定
    input_mean_vec = input_mean_vec.reshape((1,VEC_MAXLEN, VEC_SIZE))
    with graph.as_default():
        output_vec = lstm_vec_model.predict_on_batch(input_mean_vec)[0]

    ret = d2vmodel.docvecs.most_similar([output_vec])

    # 推定したベクトルから文章生成
    generated = ''
    char_IDs = [char_idx[MU] for _ in range(TXT_MAXLEN)]    #初期値は無
    rnd = random.uniform(0.1,0.5)

    for i in range(200):
        with graph.as_default():
            preds = lstm_set_model.predict_on_batch([ np.asarray([output_vec]),  np.asarray([char_IDs]) ])

        next_index = sample(preds[0], rnd)
        char_IDs = char_IDs[1:]
        char_IDs.append(next_index)
        next_char = idx_char[next_index]
        generated += next_char
        if next_char == END:
            break

    rtn_text = generated
    rtn_text = re.sub(END,'',rtn_text, flags=(re.MULTILINE | re.DOTALL))
    return rtn_text

if __name__ == '__main__':
    toots = {id.strip():toot.strip() for id,toot in zip(open("tags.txt").readlines(), open("toot_n.txt").readlines())}
    num = len(toots)
    text = ''
    while True:
        input('<press Enter>')
        rnd = random.randrange(num-VEC_MAXLEN - AVE_LEN)
        input_toots = list(toots.values())[rnd:rnd + VEC_MAXLEN + AVE_LEN]
        print('input**********************')
        print("\n".join(input_toots))
        text = lstm_gentxt(input_toots,num=1)
        print()
        print('gen text******************')
        print(text)
        print()
