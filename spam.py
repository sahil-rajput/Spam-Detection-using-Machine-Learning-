from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import zero_one_loss
import itertools
from collections import Counter
import string
import sys
import warnings
from collections import OrderedDict
from hashlib import md5
from six.moves import range
from six.moves import zip
import os
import time
import datetime
import pandas as pd
import sklearn.metrics
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from sklearn.model_selection import train_test_split

#Hyperparameters
max_features = 50000
maxlen = 400
ITERATION = 100000
batch_size = 32
embedding_dims = 100
filters = 10
kernel_size = 5
hidden_dims = 250
epochs = 2
DROPOUT = 0.1
VALIDATION_SPLIT = 0.1
HIDDEN_NODE = 4
OPTIMIZER = 'adam'


def derivative(x):
    return x*(1.0-x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
x_result = []
y_result = []


with open('data.csv') as f:
    for line in f:
        curr = line.split(',')
        new_curr = [1]
        for item in curr[:len(curr) - 1]:
            new_curr.append(float(item))
        x_result.append(new_curr)
        y_result.append([float(curr[-1])])
x_result = np.array(x_result)
x_result = preprocessing.scale(x_result)
y_result = np.array(y_result)

x_train, x_test, y_train, y_test = train_test_split(x_result, y_result, test_size=0.2)

d1 = len(x_train[0])
d2 = HIDDEN_NODE

np.random.seed(1)
w0 = 2 * np.random.random((d1, d2)) - 1
w1 = 2 * np.random.random((d2, 1)) - 1

for j in xrange(ITERATION):
    l0 = x_train
    l1 = sigmoid(np.dot(l0,w0))
    l2 = sigmoid(np.dot(l1,w1))
    l2e = y_train - l2
    l2d = l2e * derivative(l2)
    l1e = l2d.dot(w1.T)
    l1d = l1e * derivative(l1)
    w1 += l1.T.dot(l2d)
    w0 += l0.T.dot(l1d)

l0 = x_test
l1 = sigmoid(np.dot(l0,w0))
l2 = sigmoid(np.dot(l1,w1))
correct = 0
for i in xrange(len(l2)):
    if(l2[i][0] > 0.5):
        l2[i][0] = 1
    else:
        l2[i][0] = 0
    if(l2[i][0] == y_test[i][0]):
        correct += 1
print "Accuracy = ", correct * 100.0 / len(l2)
