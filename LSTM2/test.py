import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import keras.backend.tensorflow_backend as K
import tensorflow as tf
import _pickle as pk
from sklearn.metrics import mean_squared_error

sequence_length = 6

import sys

modelNAME = sys.argv[1]

model = load_model("model/"+modelNAME+'.h5')
tokenizer = pk.load(open("tokenizer", 'rb'))
X_test = np.load("data/Test_snippet.npy")
Y_test = np.load("data/Test_sentiment.npy")

X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_seq  = sequence.pad_sequences(X_test_seq,  maxlen = sequence_length)
y_pred = model.predict(X_test_seq)
#y_pred = [0 for i in range(634)]
print('MSE = ',mean_squared_error(Y_test, y_pred))

