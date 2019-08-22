'''
Setup:
    1. Past copied Python codes after import sentences bellow
    2. Set "train_result_path"
Usage:
    python ./generate_text_2LSTM_ueno.py "入学おめでとうございます．" 1000 0.5
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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, d1, c1, d2, c2, test=False):
    # Input:x -> 1
    # OneHot -> 687
    h = F.one_hot(x, (687,))

    # LSTM1 -> 200
    with nn.parameter_scope('LSTM1'):
        h = network_LSTM(h, d1, c1, 687, 100, test)

    # Slice -> 100
    h1 = F.slice(h, (0,), (100,), (1,))

    # h2:CellOut -> 100
    h2 = F.slice(h, (100,), (200,), (1,))

    # LSTM2 -> 128
    with nn.parameter_scope('LSTM2'):
        h3 = network_LSTM(h1, d2, c2, 100, 64, test)

    # h4:DelayOut
    h4 = F.identity(h1)

    # Slice_2 -> 64
    h5 = F.slice(h3, (0,), (64,), (1,))

    # h6:CellOut_2 -> 64
    h6 = F.slice(h3, (64,), (128,), (1,))

    # Affine_2 -> 687
    h7 = PF.affine(h5, (687,), name='Affine_2')

    # h8:DelayOut_2
    h8 = F.identity(h5)
    # h7:Softmax
    h7 = F.softmax(h7)
    return h2, h4, h6, h8, h7
    # h2:CellOut -> 100
    # h4:DelayOut
    # h6:CellOut_2 -> 64
    # h8:DelayOut_2
    # h7:Softmax

def network_LSTM(x, D, C, InputShape, HiddenSize, test=False):
    # Input_2:x -> 687
    # Delya_in:D -> 100
    # Cell_in:C -> 100

    # Concatenate -> 787
    h = F.concatenate(D, x, axis=1)

    # Affine -> 100
    h1 = PF.affine(h, HiddenSize, name='Affine')

    # InputGate -> 100
    h2 = PF.affine(h, HiddenSize, name='InputGate')

    # OutputGate -> 100
    h3 = PF.affine(h, HiddenSize, name='OutputGate')

    # ForgetGate -> 100
    h4 = PF.affine(h, HiddenSize, name='ForgetGate')
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
    # Dropout
    if not test:
        h5 = F.dropout(h5)

    # Output
    h5 = F.identity(h5)

    # Concatenate_2 -> 200
    h5 = F.concatenate(h5, h6, axis=1)
    return h5


train_result_path = "C:/Users/yoshidh/WorkData/NNC/0724UenoW.files/20190815_124958" + "/results.nnp"
# train_result_path = "./result_train.nnp"
hidden_size1 = 100
hidden_size2 = 64
char_table_path = "./UenoTodaiSpeech+.pkl"

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
    nn.load_parameters(train_result_path)
    x = nn.Variable((1,1))
    d1 = nn.Variable((1,hidden_size1))
    c1 = nn.Variable((1,hidden_size1))
    d2 = nn.Variable((1,hidden_size2))
    c2 = nn.Variable((1,hidden_size2))
    CO1, DO1, CO2, DO2, SM = network(x, d1, c1, d2, c2, test=True)
    # h2:CellOut -> 100
    # h4:DelayOut
    # h6:CellOut_2 -> 64
    # h8:DelayOut_2
    # h7:Softmax
  
    d1.d = np.random.rand(1,hidden_size1)
    c1.d = np.random.rand(1,hidden_size1)
    d2.d = np.random.rand(1,hidden_size2)
    c2.d = np.random.rand(1,hidden_size2)
    for c in first_word:
        x.d[0] = char_table.str2ids(c)[0]
        CO1.forward()
        DO1.forward()
        CO2.forward()
        DO2.forward()
        SM.forward()
        dist = SM.d[0]
        d1.d = DO1.d
        c1.d = CO1.d
        d2.d = DO2.d
        c2.d = CO2.d
        print(c,end="")

    id = sample(dist,temperature)
    for i in range(text_size-len(first_word)):
        x.d[0] = id
        CO1.forward()
        DO1.forward()
        CO2.forward()
        DO2.forward()
        SM.forward()
        dist = SM.d[0]
        d1.d = DO1.d
        c1.d = CO1.d
        d2.d = DO2.d
        c2.d = CO2.d

        id = sample(dist,temperature)
        char = char_table.id2char(id)
        print(char,end="")
    print()

if __name__ == '__main__':
    main()


