# -*- coding: utf-8 -*-

# from gensim.models import word2vec
import logging
import sys,os
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 空のリストを作成（学習データとなる各文書を格納）

f = open('tmp/ids_merge.txt', 'r')
tags = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
f.close()

f = open('tmp/wakati.txt', 'r')
lines = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
f.close()

training_docs = []
sentents = []
for (line,tag) in zip(lines,tags):
    # 各文書を表すTaggedDocumentクラスのインスタンスを作成
    # words：文書に含まれる単語のリスト（単語の重複あり）
    # tags：文書の識別子（リストで指定．1つの文書に複数のタグを付与できる）
    sentents.append(line)
    sent = TaggedDocument(words=line.split(), tags=tag.split())
    # 各TaggedDocumentをリストに格納
    training_docs.append(sent)

# 学習実行（パラメータを調整可能）
# documents:学習データ（TaggedDocumentのリスト）
# min_count=1:最低1回出現した単語を学習に使用する
# 学習モデル=DBOW（デフォルトはdm=1:学習モデル=DM）
if not os.path.exists('out/d2v.model'):
    model = Doc2Vec(documents=training_docs,
                    vector_size=32,
                    window=8,
                    # alpha=0.025,
                    # min_alpha=0.0001,
                    min_count=10,
                    # sample=1e-5,
                    # workers=2,
                    epochs=50,
                    # negative=5,
                    # hs=0,
                    # dm=1,
                    # dbow_words=0,
                    )
else:
    model = Doc2Vec.load('out/d2v.model')
    model.train(documents=training_docs, epochs=50, total_examples=model.corpus_count)

#model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=False)
model.save('out/d2v.model')
