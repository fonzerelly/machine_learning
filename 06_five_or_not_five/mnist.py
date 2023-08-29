import sys
import numpy as np
import gzip
import struct

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
    return np.insert(X, 0,1, axis=1)

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        all_labels = f.read()
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1,1)
    
def encode_fives(Y):
    return (Y == 5).astype(int)



X_train = prepend_bias(load_images('/usr/src/mnist/train-images-idx3-ubyte.gz'))

X_test = prepend_bias(load_images('/usr/src/mnist/t10k-images-idx3-ubyte.gz'))

Y_train_unencoded = load_labels('/usr/src/mnist/train-labels-idx1-ubyte.gz')

Y_test_unencoded = load_labels('/usr/src/mnist/t10k-labels-idx1-ubyte.gz')

Y_train = encode_fives(Y_train_unencoded)
Y_test = encode_fives(Y_test_unencoded)


def printDigit(digit):
    inChars = list(map(lambda x: do(x),digit))
    for i, c in enumerate(inChars):
        if (i % 28 == 0):
            print (c)
        else:
            print (c, end='')

def do(x):
    if (x < 240):
        return '.'
    return '@'

if __name__ == "__main__":
    id = int(sys.argv[1])

    X_print = X_test
    Y_print = Y_test_unencoded
    if (len(sys.argv) > 2 and sys.argv[2] == 'train'):
        X_print = X_train
        Y_print = Y_train_unencoded
    print('Label: %s' % Y_print[id][0])
    printDigit(X_print[id])
