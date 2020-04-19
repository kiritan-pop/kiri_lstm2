# coding: utf-8

import sys
import io,re,random
import unicodedata
from multiprocessing import Process, Queue

pat1 = re.compile(r'《.*》|\||｜|［.*］|\[.*\]|^.\n')
pat3 = re.compile(r'^\n')
pat4 = re.compile(r'^.\n')
pat5 = re.compile(r'([。、…])\1+')
pat6 = re.compile(r'(.)\1{4,}')
WORKERS = 3
timeout = 5
BUF_LINES = 50000
EOF_SW = False
NRM_SW = []
un_func = unicodedata.normalize

def normalize(num,readQ,writeQ):
    print('--Process(%d) start' %(num) )
    while True:
        try:
            lines = readQ.get(timeout=timeout) #キューからトゥートを取り出すよー！
            outs = []
            for line in lines:
                line = pat1.sub('',line)
                text = line.strip() + '\n'
                outs.append(text)

            writeQ.put(outs)
        except:
            return

def reader(readQ):
    lines = []
    print('--Start reading--')
    for i,line in enumerate(open('tmp/toot_merge.txt', 'r')):
        lines.append(line)
        if BUF_LINES > 0 and i % BUF_LINES == BUF_LINES - 1:
            readQ.put(lines)
            lines = []
    #あまりがあれば
    if len(lines) > 0:
        readQ.put(lines)

    print('--Finish reading--')

def writer(writeQ):
    with open('tmp/toot_merge_n.txt', 'w') as fo:
        while True:
            try:
                lines = writeQ.get(timeout=timeout)
                print('--Process(writer)::lines %d ' %(len(lines)) )
                fo.write("".join(lines))
            except:
                return
    print('--Finish writing--')

if __name__ == '__main__':
    readQ = Queue()
    writeQ = Queue()

    p_r = Process(target=reader, args=(readQ,))
    p_r.start()

    p_n = []
    for num in range(0,WORKERS-2):
        tmp  = Process(target=normalize, args=(num,readQ,writeQ))
        tmp.start()
        p_n.append(tmp)
    p_w = Process(target=writer, args=(writeQ,))
    p_w.start()

    p_r.join()
    for p_n_a in p_n :
        p_n_a.join()
    p_w.join()
