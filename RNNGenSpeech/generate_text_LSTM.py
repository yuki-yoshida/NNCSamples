'''
Setup:
    1. Past copied Python codes after import sentences bellow
    2. Set "train_result_path"
Usage:
    python ./generate_text_LSTM.py "first_word" length temp
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

def network(x, D, C, test=False):
    # Input:x -> 1
    # Delay_in:D -> 100
    # Cell_in:C -> 100
    # OneHot -> 38
    h = F.one_hot(x, (38,))

    # Concatenate -> 138
    h = F.concatenate(h, D, axis=1)

    # Affine -> 100
    h1 = PF.affine(h, (100,), name='Affine')

    # InputGate -> 100
    h2 = PF.affine(h, (100,), name='InputGate')

    # OutputGate -> 100
    h3 = PF.affine(h, (100,), name='OutputGate')

    # ForgetGate -> 100
    h4 = PF.affine(h, (100,), name='ForgetGate')
    # Sigmoid
    h1 = F.sigmoid(h1)
    # Sigmoid_2
    h2 = F.sigmoid(h2)

    # Sigmoid_3
    h3 = F.sigmoid(h3)
    # Sigmoid_4
    h4 = F.sigmoid(h4)

    # Mul2 -> 100
    h1 = F.mul2(h1, h2)

    # Mul2_3 -> 100
    h4 = F.mul2(h4, C)

    # Add2 -> 100
    h1 = F.add2(h1, h4, True)

    # Tanh
    h5 = F.tanh(h1)

    # Cell_out
    h6 = F.identity(h1)

    # Mul2_2 -> 100
    h5 = F.mul2(h5, h3)

    # Affine_2 -> 38
    h7 = PF.affine(h5, (38,), name='Affine_2')

    # Delay_out
    h8 = F.identity(h5)
    # Softmax
    h7 = F.softmax(h7)
    return h6, h8, h7

train_result_path = "C:/Users/yoshidh/WorkData/NNC/0809Obama20W.files/20190809_144208"
hidden_size = 100
char_table_path = "./ObamaHiroshimaSpeech.pkl"

tweak1 = 0.00001 # to force pred > 0
tweak2 = 1.00001 # to force sum(pred) < 1

def sample(preds, temperature):
    if temperature == 0:
        return np.argmax(preds)
    preds = np.log(preds + tweak1) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds / tweak2, 1)
    return np.argmax(probas)

def main():
    first_word = sys.argv[1].lower()
    # first_word = " おめでとうございます" 
    text_size = int(sys.argv[2])
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    with open(char_table_path, "rb") as f:
        char_table = pickle.load(f)
    nn.load_parameters(train_result_path+"/results.nnp")
    x = nn.Variable((1,1))
    d = nn.Variable((1,hidden_size))
    c = nn.Variable((1,hidden_size))
    h7, h9, h8 = network(x, d, c, test=True)
  
    d.d = np.random.rand(1,hidden_size)
    c.d = np.random.rand(1,hidden_size)
    for char in first_word:
        x.d[0] = char_table.str2ids(char)[0]
        h7.forward()
        h9.forward()
        h8.forward()
        dist = h8.d[0]
        d.d = h9.d
        c.d = h7.d
        print(char,end="")

    sample_and_print = lambda id: print(char_table.id2char(id),end="")

    id = sample(dist,temperature)
    sample_and_print(id)
    for i in range(text_size-len(first_word)):
        x.d[0] = id
        h7.forward()
        h9.forward()
        h8.forward()
        dist = h8.d[0]
        d.d = h9.d
        c.d = h7.d

        id = sample(dist,temperature)
        sample_and_print(id)
    print()

if __name__ == '__main__':
    main()


