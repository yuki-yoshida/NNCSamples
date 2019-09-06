'''
[参考] KerasのSingle-LSTM文字生成サンプルコードを解説
https://qiita.com/YankeeDeltaBravo225/items/487dbfa1bef02bcfb84c
Usage:
   python ./make_dict.py ./ObamaHiroshimaSpeech.txt ./ObamaHiroshimaSpeech.pkl

'''

from __future__ import print_function

import numpy as np
import sys
import io
import pickle
import unicodedata

class CharTable:
    def __init__(self, text):
        uniq_chars = sorted(list(set(text)))
        self.id_num = len(uniq_chars)
        self.char2id_dict = dict((c, i) for i, c in enumerate(uniq_chars))
        self.id2char_dict = dict((i, c) for i, c in enumerate(uniq_chars))


    def str2ids(self, str):
        return [ self.char2id_dict[char] for char in str]


    def ids2str(self, ids):
        return ''.join([ self.id2char_dict[id] for id in ids])

    def id2char(self, id):
        return self.id2char_dict[id]

    '''
    def char2id(self, char):
        return self.char2id_dict[char]
    '''

def is_japanese(string):
    for ch in string:
        name = unicodedata.name(ch) 
        if "CJK UNIFIED" in name \
        or "HIRAGANA" in name \
        or "KATAKANA" in name:
            return True
    return False

def load_text(path):
    with io.open(path, encoding='utf-8') as f:
        raw_text = f.read()
    
    if is_japanese(raw_text[0:10]): # 先頭10文字に日本語があったら日本語
        # 日本語の時は、改行を無くし、半角を全角に変更、「。、」は「．，」に統一
        text = ''.join(raw_text.splitlines())
        HAN = "".join(chr(0x21 + i) for i in range(94)) + " 。、"
        ZEN = "".join(chr(0xff01 + i) for i in range(94)) + "　．，"
        HAN2ZEN = str.maketrans(HAN, ZEN)
        text = text.translate(HAN2ZEN)
    else:
        # 英語の時は、改行を空白にし、小文字に変更。
        text = ' '.join(raw_text.splitlines()).lower()
    print('text length:', len(text))
    return text

def make_dict(text,path):
    char_table = CharTable(text)
    print('code size:', len(char_table.char2id_dict))
    with open(path, "wb") as f:
        pickle.dump(char_table, f)

def load_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)
