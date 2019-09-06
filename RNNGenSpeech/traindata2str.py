'''
Usage:
    python ./traindata2str.py ./ObamaHiroshimaSpeech.pkl ./OHS_test.csv 5
'''

from __future__ import print_function

from make_dict import CharTable, load_dict
import sys
import io
import csv

def traindata2str(path, char_table, row1, row2):
    with io.open(path) as f:
        mat = list(csv.reader(f))
    seq_len = int(len(mat[0])) + 1
    mat = mat[1:]
    for i in range(row1,row2+1):
        print(str(i)+":",end="")
        row = [int(elm) for elm in mat[i][0:seq_len]]
        print(row)
        print(char_table.ids2str(row))
            
def main():
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    row1 = int(sys.argv[3])
    row2 = int(sys.argv[4]) if len(sys.argv) > 4 else row1
    char_table = load_dict(path1)
    print('code size:', len(char_table.char2id_dict))
    traindata2str(path2, char_table, row1, row2)

if __name__ == '__main__':
    main()