'''
Usage:
   python ./create_RNNtrain_data.py ./ObamaHiroshimaSpeech.txt 20 OHS 10
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
import sys
import os
import io
import csv
import random

def make_data(text_ids, seq_len):
    last_start_idx = len(text_ids) - seq_len - 1
    lines = []
    for k in range(0, last_start_idx):
        line = [text_ids[i] for i in range(k, k+seq_len)]
        line += [text_ids[k+seq_len]]
        lines.append(line)
    random.shuffle(lines)
    return lines

def make_headline(seq_len):
    headline = ["x__" + str(i) for i in range(0, seq_len)]
    headline += "y"
    return headline
    
def create_train_data(lines, headline, header, train_size):
    path_train = "./" + header + "_train.csv"
    with open(path_train, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(headline)
        writer.writerows(lines[:train_size])
    print(path_train,"is generated whose length is", train_size)

    path_test = "./" + header + "_test.csv"
    with open(path_test, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(headline)
        writer.writerows(lines[train_size:])
    print(path_test,"is generated whose length is", len(lines)-train_size)

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
    lines = make_data(text_ids,seq_len)
    headline = make_headline(seq_len)
    create_train_data(lines, headline, header, train_size)

if __name__ == '__main__':
    main()