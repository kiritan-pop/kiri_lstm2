# -*- coding: utf-8 -*-

import MeCab
import sys
from gensim.models.doc2vec import Doc2Vec

tagger = MeCab.Tagger('-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd -u ../dic/nicodic.dic')
model   = Doc2Vec.load(sys.argv[1])
tags = [tmp.strip() for tmp in open(sys.argv[2]).readlines()]
toots = {tag:toot.strip() for tag,toot in zip(tags,open(sys.argv[3]).readlines())}

while True:
    texts = input(">>").strip()
    texts = tagger.parse(texts).split(" ")
    print(texts)
    vec = model.infer_vector(texts)
    ret = model.docvecs.most_similar([vec])
    for id, score in ret:
        print(toots[id])
