# -*- coding: utf-8 -*-

import MeCab
import sys,re

tagger = MeCab.Tagger('-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd -u ../kiri_bot/dic/nicodic.dic')
# tagger = MeCab.Tagger('-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd -u ../dic/name.dic,../dic/id.dic,../dic/nicodic.dic')

fo = open(sys.argv[2], 'w')
i = 0

for line in open(sys.argv[1], 'r'):
    result = tagger.parse(line)
    if result != None:
        fo.write(result)
        if i % 100000 == 0:
            print("i:",i)
        i += 1
    else:
        print(result)

fo.close()
