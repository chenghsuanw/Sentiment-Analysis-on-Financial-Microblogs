import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense,Bidirectional,LSTM,GRU, Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
import _pickle as pk

import keras.backend.tensorflow_backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json

import sys

#total 1579 snippets
#average length of snippets is 5.157061431285624
#maximum length of snippet is 21

max_voc_count = 3000
sequence_length = 6
word_embedd = 300
hidden_size = 128
hidden_size2 = 128


modelNAME = sys.argv[1]


def	get_embedding(word_index):
	embeddings_index = {}
	f = open("data/NTUSD-Fin/NTUSD_Fin_word_v1.0.json","r")
	wordlist  = json.load(f)
	for w in wordlist:
		word = w["token"]
		coefs = w["word_vec"]
		embeddings_index[word] = coefs
	f.close()

	embedding_matrix = np.zeros((max_voc_count, word_embedd))
	for word, i in word_index.items():
		if i < max_voc_count :
			embedding_vector = embeddings_index.get(word)  #if word not in dict, return None
			if embedding_vector is not None:
				 # words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector
	return embedding_matrix

def buildModel(embedding_matrix):
	model = Sequential()
	model.add(Embedding (input_dim = max_voc_count, output_dim = word_embedd, input_length = sequence_length, weights=[embedding_matrix] ))
	model.add(Bidirectional(GRU(hidden_size,dropout=0.5,recurrent_dropout=0.5)))
	#model.add(Bidirectional(GRU(hidden_size,dropout=0.5,recurrent_dropout=0.25,return_sequences=True)))
	#model.add(GRU(hidden_size2,dropout=0.5,recurrent_dropout=0.25))

	#model.add(Bidirectional(GRU(hidden_size,return_sequences=True)))
	#model.add(GRU(hidden_size2))
	model.add(Dense(1))
	Adam = optimizers.adam(lr=1e-3)
	model.compile(loss='mean_squared_error', optimizer= Adam )
	return model

def main():
	X_all = np.load("data/snippet.npy")
	Y_all = np.load("data/sentiment.npy")
	X_train, X_valid, Y_train, Y_valid  = X_all[120:],X_all[0:120], Y_all[120:],Y_all[0:120]

	tokenizer = Tokenizer(max_voc_count,filters='\n\t')
	tokenizer.fit_on_texts(X_all)
	pk.dump(tokenizer, open("tokenizer", 'wb')) #save the tokenizer

	###text to seq and pad###
	X_all_seq = tokenizer.texts_to_sequences(X_all)
	X_train_seq = tokenizer.texts_to_sequences(X_train)
	X_valid_seq = tokenizer.texts_to_sequences(X_valid)

	X_all_seq = sequence.pad_sequences(X_all_seq, maxlen = sequence_length)
	X_train_seq = sequence.pad_sequences(X_train_seq, maxlen = sequence_length)
	X_valid_seq = sequence.pad_sequences(X_valid_seq, maxlen = sequence_length)

	word_index = tokenizer.word_index
	embedding_matrix = get_embedding(word_index)
	'''	
	###count how many word are in NTUSD
	count = 0 
	for i  in range(embedding_matrix.shape[0]):
		if embedding_matrix[i][0] != 0:
			count +=1
	print(count)
	print("-------------------------------------")
	### 1312
	'''
	model = buildModel(embedding_matrix)
	#print(model.summary())

	#history = model.fit( X_train_seq, Y_train, validation_data = (X_valid_seq,Y_valid), epochs= 23 )
	history = model.fit( X_all_seq, Y_all, epochs= 23 )
	model.save('model/'+modelNAME+'.h5')	

if __name__ == '__main__':
	main()
