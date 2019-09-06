'''
Usage:
   python ./create_RNNtrain_dataW.py ./ObamaHiroshimaSpeech.txt 20 OHSW 10
   where
    arg1: input text file
    arg2: sequence length
    arg3: head string for training dataset files
    arg4: % of test data (default is 10%)
Output:
    ./ObamaHiroshimaSpeech.pkl: dumped dictionary
    ./OHS_train.csv: train dataset
    ./OHS_test.csv: test dataset
'''

from __future__ import print_function

import make_dict as md
import create_RNNtrain_data as crRNN
import sys
import os
import io
import csv
import random

def make_dataW(text_ids, seq_len):
    last_start_idx = len(text_ids) - seq_len - 1
    lines = []
    for k in range(0, last_start_idx):
        linex = [text_ids[i] for i in range(k, k+seq_len)]
        liney = [text_ids[i] for i in range(k+1, k+seq_len+1)]
        lines.append(linex + liney)
    random.shuffle(lines)
    return lines

def make_headlineW(seq_len):
    headline = ["x__" + str(i) for i in range(0, seq_len)]
    headline += ["y__" + str(i) for i in range(0, seq_len)]
    return headline
    
def main():
    text_path = sys.argv[1]
    seq_len = int(sys.argv[2])
    header = sys.argv[3]
    test_size = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    text = md.load_text(text_path)
    train_size = int((len(text) - seq_len - 1) * (100 - test_size) / 100)
    dict_path = os.path.splitext(text_path)[0] + ".pkl"
    md.make_dict(text,dict_path)
    char_table = md.load_dict(dict_path)
    
    text_ids = char_table.str2ids(text)
    lines = make_dataW(text_ids,seq_len)
    headline = make_headlineW(seq_len)
    crRNN.create_train_data(lines, headline, header, train_size)

if __name__ == '__main__':
    main()