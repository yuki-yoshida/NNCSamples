'''
Setup:
    1. Past copied Python codes after import sentences bellow
    2. Set "train_result_path"
Usage:
    python ./add_digits.py 314159265359 271828182846
'''
from __future__ import print_function
from make_dict import CharTable
import numpy as np
import sys


train_result_path = "C:"
hidden_size = 16

def main():
    n1 = sys.argv[1]
    n2 = sys.argv[2]
    size = max(len(n1),len(n2))
    zs = "0" * size
    n1 = (zs + n1)[len(n1):]
    n2 = (zs + n2)[len(n2):]
    num1 = np.array([int(i) for i in reversed(list(n1))])
    num2 = np.array([int(i) for i in reversed(list(n2))])
    num3 = np.zeros(size,dtype=int)
    num4 = np.zeros(size,dtype=int)
    n = 0
    for i in range(size):
        n, num4[i] = divmod(num1[i] + num2[i] + n,10)

    nn.load_parameters(train_result_path+"/results.nnp")
    x1 = nn.Variable((1,1))
    x2 = nn.Variable((1,1))
    d = nn.Variable((1,hidden_size))
    h3, h2 = network(x1, x2, d, test=True)

    d.d = np.zeros((1,hidden_size))
    for i in range(size):
        x1.d[0] = num1[i]
        x2.d[0] = num2[i]
        h3.forward()
        h2.forward()
        d.d = h3.d
        num3[i] = np.ndarray.argmax(h2.d)

    print("   ","".join([str(i) for i in reversed(num1)]))
    print("  +","".join([str(i) for i in reversed(num2)]))
    print("  =","".join([str(i) for i in reversed(num3)]))
    print("( =","".join([str(i) for i in reversed(num4)]),")")

if __name__ == '__main__':
    main()


