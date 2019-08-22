'''
Setup:
    1. Past copied Python codes after import sentences bellow
    2. Set "train_result_path"
Usage:
    python ./generate_text.py "first_word" length temp
'''
from __future__ import print_function
from make_dict import CharTable
import numpy as np
import pickle
import sys
import re

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, D, test=False):
    # Input:x -> 1
    # Delay_in:D -> 100
    # OneHot -> 38
    h = F.one_hot(x, (38,))

    # Concatenate -> 138
    h = F.concatenate(h, D, axis=1)
    # Affine -> 100
    h = PF.affine(h, (100,), name='Affine')
    # Sigmoid
    h = F.sigmoid(h)

    # Affine_2 -> 38
    h1 = PF.affine(h, (38,), name='Affine_2')

    # Delay_out
    h2 = F.identity(h)
    # Softmax
    h1 = F.softmax(h1)
    return h2, h1


train_result_path = "C:/Users/yoshidh/WorkData/NNC/0806Obama.files/20190806_172002"
hidden_size = 100
char_table_path = "./ObamaHiroshimaSpeech.pkl"
skip_len = 10

tweak = 1.00001 # to force sum(pred) < 1.0

def sample(preds, temperature):
    if temperature == 0:
        return np.argmax(preds)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds / tweak, 1)
    return np.argmax(probas)

def main():
    first_word = " "*skip_len + sys.argv[1].lower()
    first_word = first_word[len(first_word) - skip_len:]
    text_size = int(sys.argv[2])
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0

    with open(char_table_path, "rb") as f:
        char_table = pickle.load(f)
    nn.load_parameters(train_result_path+"/results.nnp")
    x = nn.Variable((1,1))
    d = nn.Variable((1,hidden_size))
  
    d.d = np.random.rand(1,hidden_size)
    for i in range(skip_len):
        x.d[0] = char_table.str2ids(first_word[i])[0]
        delayo, softm = network(x, d, test=True)
        delayo.forward()
        softm.forward()
        dist = softm.d[0]
        d.d = delayo.d
        print(first_word[i],end="")

    id = sample(dist,temperature)
    for i in range(text_size-skip_len):
        x.d[0] = id
        delayo, softm = network(x, d)
        delayo.forward()
        softm.forward()
        dist = softm.d[0]
        d.d = delayo.d
        id = sample(dist,temperature)
        char = char_table.id2char(id)
        print(char,end="")
    print()

if __name__ == '__main__':
    main()


