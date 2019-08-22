'''
Usage:
    python ./show_dict.py ./ObamaHiroshimaSpeech.pkl
'''

from __future__ import print_function

from make_dict import CharTable
import pickle
import sys
import io

def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        char_table = pickle.load(f)
    print(char_table.ids2str(range(0,len(char_table.char2id_dict))))
    print('code size:', len(char_table.char2id_dict))


if __name__ == '__main__':
    main()