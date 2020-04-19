# -*- coding: utf-8 -*-

import MeCab
import sys,re

tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/')

fo = open('tmp/wakati.txt', 'w')
i = 0

for line in open('tmp/toot_merge_n.txt', 'r'):
    result = tagger.parse(line)
    if result != None:
        fo.write(result)
        if i % 100000 == 0:
            print("i:",i)
        i += 1
    else:
        print(result)

fo.close()
