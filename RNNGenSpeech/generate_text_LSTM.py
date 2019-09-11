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

train_result_path = "Your trained result nnp"
hidden_size = 100
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


