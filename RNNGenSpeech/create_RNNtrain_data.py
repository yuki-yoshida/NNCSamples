'''
Usage:
   python ./create_train_data.py ./ObamaHiroshimaSpeech.txt 10 OHS 7000
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
        line = [text_ids[i] for i in range(k, k+seq_len)]
        line += [text_ids[k+seq_len]]
        lines.append(line)
    random.shuffle(lines)

    header = ["x__" + str(i) for i in range(0, seq_len)]
    header += "y"
    
    path_train = "./" + head + "_train.csv"
    with open(path_train, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(lines[:train_size])

    path_test = "./" + head + "_test.csv"
    with open(path_test, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(lines[train_size:])

def main():
    argc = len(sys.argv)
    path = sys.argv[1]
    seq_len = int(sys.argv[2])
    head = sys.argv[3]
    train_size = int(sys.argv[4])
    text = md.load_text(path)
    print('text length:', len(text))
    if len(text) * 0.9 < train_size:
        print('train size is too large:', train_size)
        return
    char_table = md.CharTable(text)
    print('code size:', len(char_table.char2id_dict))
    text_ids = char_table.str2ids(text)
    create_train_data(text_ids, seq_len, head, train_size)

if __name__ == '__main__':
    main()