import numpy as np
import keras
from keras.preprocessing.text import Tokenizer

X_all = np.load("data/snippet.npy")
#print(type(X_all))


each_S_len = [len(string.split(" ")) for string in X_all]
print("total %d snippets" %len(each_S_len) )
print("average length of snippets is",sum(each_S_len) / len(each_S_len))
print("maximum length of snippet is", max(each_S_len) )


'''
max_voc_count = 3000
tokenizer = Tokenizer(max_voc_count,filters='\n\t')
tokenizer.fit_on_texts(X_all)
seq2 = tokenizer.texts_to_sequences(["to the a in"])
print(tokenizer.word_index)
print(seq2)
'''
