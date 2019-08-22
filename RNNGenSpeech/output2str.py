'''
Usage:
    python ./output2str.py ./ObamaHiroshimaSpeech.pkl ../0612Obama.files/20190617_113554/output_result.csv 0 5
'''

from __future__ import print_function

from make_dict import CharTable
import numpy as np
import pickle
import sys
import io
import csv

def output2str(path, char_table, row1, row2, tops):
    with io.open(path) as f:
        mat = list(csv.reader(f))
    row0 = mat[0]
    dict_size = int(row0[len(row0)-1][4:])+1
    data_len = len(row0) - dict_size - 1
    mat = mat[1:]
    for i in range(row1,row2+1):
        xs = [int(elm) for elm in mat[i][0:data_len]]
        y = int(mat[i][data_len])
        dist = [float(elm) for elm in mat[i][data_len+1:]]
        print("x:",char_table.ids2str(xs),end=" ")
        print("y:",char_table.ids2str([y]),"y':",end=" ")
        for k in range(tops):
            ans = np.argmax(dist) 
            print(char_table.ids2str([ans]),end="")
            print("*" if y == ans else " ",end="")
            dist[ans] = 0.0
        print()
            
def main():
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    row1 = int(sys.argv[3])
    row2 = int(sys.argv[4]) if len(sys.argv) > 4 else row1
    tops = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    with open(path1, "rb") as f:
        char_table = pickle.load(f)
    print('code size:', len(char_table.char2id_dict))
    output2str(path2, char_table, row1, row2, tops)

if __name__ == '__main__':
    main()