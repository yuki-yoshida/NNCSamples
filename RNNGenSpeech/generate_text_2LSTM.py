'''
Setup:
    1. Past copied Python codes after import sentences bellow
    2. Set "train_result_path"
Usage:
    python ./generate_text_2LSTM.py "first_word" length temp
'''
from __future__ import print_function
from make_dict import CharTable
import numpy as np
import pickle
import sys
import re

train_result_path = "Your trained result nnp"
hidden_size1 = 100
hidden_size2 = 64
char_table_path = "./UenoTodaiSpeech+.pkl"

def sample(preds, temperature):
    tweak1 = 0.00001 # to force pred > 0
    tweak2 = 1.00001 # to force sum(pred) < 1

    if temperature == 0:
        return np.argmax(preds)
    preds = np.log(preds + tweak1) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds / tweak2, 1)
    return np.argmax(probas)

def main():
    first_word = sys.argv[1].lower()
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

    sample_and_print = lambda id: print(char_table.id2char(id),end="")

    id = sample(dist,temperature)
    sample_and_print(id)
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
        sample_and_print(id)
    print()

if __name__ == '__main__':
    main()


