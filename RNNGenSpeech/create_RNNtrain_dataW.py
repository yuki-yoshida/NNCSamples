'''
Usage:
   python ./create_RNNtrain_dataW.py ./ObamaHiroshimaSpeech.txt 20 OHS 10
   where
    arg1: input text file
    arg2: sequence length
    arg3: head string for training dataset files
    arg4: % of test data (default is 10%)
'''

from __future__ import print_function

import make_dict as md
import sys
import io
import csv
import random

def create_train_data(text_ids, seq_len, head, train_size):
    last_start_idx = len(text_ids) - seq_len - 1
    lines = []
    for k in range(0, last_start_idx):
        linex = [text_ids[i] for i in range(k, k+seq_len)]
        liney = [text_ids[i] for i in range(k+1, k+seq_len+1)]
        lines.append(linex + liney)
    random.shuffle(lines)

    header = ["x__" + str(i) for i in range(0, seq_len)]
    header += ["y__" + str(i) for i in range(0, seq_len)]
    
    path_train = "./" + head + "_train.csv"
    with open(path_train, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(lines[:train_size])
    print(path_train,"is generated whose length is", train_size)

    path_test = "./" + head + "_test.csv"
    with open(path_test, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(lines[train_size:])
    print(path_test,"is generated whose length is", last_start_idx-train_size)

def main():
    path = sys.argv[1]
    seq_len = int(sys.argv[2])
    head = sys.argv[3]
    test_size = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    text = md.load_text(path)
    print('text length:', len(text))
    train_size = int((len(text) - seq_len - 1) * (100 - test_size) / 100)
    char_table = md.CharTable(text)
    print('code size:', len(char_table.char2id_dict))
    text_ids = char_table.str2ids(text)
    create_train_data(text_ids, seq_len, head, train_size)

if __name__ == '__main__':
    main()