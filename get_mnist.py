import io
import numpy as np
import gzip
import tarfile
import pickle

def fetch_mnist():
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  X_train = parse("datasets/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse("datasets/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse("datasets/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse("datasets/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test


def fetch_fashionmnist():
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  X_train = parse("datasets/fashionmnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse("datasets/fashionmnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse("datasets/fashionmnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse("datasets/fashionmnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test



