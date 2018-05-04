import numpy as np
import tensorflow as tf
import json
import pickle
import os
import re
from sklearn.metrics import f1_score
from model import *

class Twit(object):
	def __init__(self, tweet, target, snippet, sentiment):
		self.tweet = tweet
		self.target = target
		self.snippet = snippet
		self.sentiment = sentiment
		
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('training_data', default='./training_set.json', help='training set path')
tf.app.flags.DEFINE_string('testing_data', default='./test_set.json', help='testing set path')
tf.app.flags.DEFINE_string('model_path', default='./model/model.ckpt', help='model path')

tf.app.flags.DEFINE_string('w2v', default='./NTUSD_Fin_word_v1.0.json', help='pretrained word to vector')
tf.app.flags.DEFINE_integer('word_embed_dim', default=300, help='word embedding dimension')

tf.app.flags.DEFINE_integer('rnn_units', default=256, help='number of hidden units of rnn cell')
tf.app.flags.DEFINE_integer('hidden_layer', default=128, help='number of hidden units of fully connected layer')
tf.app.flags.DEFINE_integer('max_length', default=10, help='max length of the sequence')
tf.app.flags.DEFINE_integer('epochs', default=100, help='epochs when training')
tf.app.flags.DEFINE_integer('batch_size', default=100, help='batch size per iteration')
tf.app.flags.DEFINE_float('lr_rate', default=0.001, help='learning rate')


special_tokens = ['<UNK>', '<PAD>']

def load_data(path):
	print('Loading data...')

	datas = json.load(open(path))
	data = []

	for twit in datas:
		tweet = twit['tweet']
		target = twit['target']
		if type(twit['snippet']) == type(str()):
			snippet = re.sub(r'[^\w\s]', '', twit['snippet'].lower())
		else:
			snippet = ' '.join(twit['snippet'])
			snippet = re.sub(r'[^\w\s]', '', snippet.lower())
		sentiment = float(twit['sentiment'])
		data.append(Twit(tweet, target, snippet, sentiment))
	
	print('There are {} datas.'.format(len(data)))
	
	return data

def build_word_vector(w2v_path, word_embed_dim):
	print('Building dictionary and word vectors...')

	w2i, w2v = dict(), []
	index = 0

	for word in special_tokens:
		w2v.append(np.random.normal(0, 0.1, word_embed_dim).astype(np.float32))
		w2i[word] = index
		index += 1

	content = json.load(open(w2v_path))
	for word in content:
		w2v.append(np.array(word['word_vec'], dtype=np.float32))
		w2i[word['token'].lower()] = index
		index += 1

	w2v = np.array(w2v)

	with open('w2i.pickle', 'wb') as f:
		pickle.dump(w2i, f)
	with open('w2v.pickle', 'wb') as f:
		pickle.dump(w2v, f)

	print('There are {} word vectors. Each vector has {} dimension.'.format(len(w2v), word_embed_dim))

def tokenize(string, w2i, max_length):
	index_list = []
	str_list = string.split()

	for i, word in enumerate(str_list):
		if i < max_length-1:
			if word in w2i:
				index_list.append(w2i[word])
			else:
				index_list.append(w2i['<UNK>'])
		else:
			break

	while len(index_list) < max_length:
		index_list.append(w2i['<PAD>'])

	return np.array(index_list)

def main():
	
	training_data = load_data(FLAGS.training_data)
	testing_data = load_data(FLAGS.testing_data)

	if not os.path.exists('w2i.pickle') or not os.path.exists('w2v.pickle'):
		build_word_vector(FLAGS.w2v, FLAGS.word_embed_dim)
	with open('w2i.pickle', 'rb') as f:
		w2i = pickle.load(f)
	with open('w2v.pickle', 'rb') as f:
		w2v = pickle.load(f)

	input_index = np.array([tokenize(twit.snippet, w2i, FLAGS.max_length) for twit in training_data])
	labels = np.array([twit.sentiment for twit in training_data])

	test_input_index = np.array([tokenize(twit.snippet, w2i, FLAGS.max_length) for twit in testing_data])
	test_labels = np.array([twit.sentiment for twit in testing_data])

	model = LSTM(FLAGS, w2v)
	model.train(input_index, labels, FLAGS.model_path)
	output, loss = model.test(test_input_index, test_labels, FLAGS.model_path)

	truth, predict = [], []

	for twit in testing_data:
		if twit.sentiment > 0:
			truth.append(1)
		elif twit.sentiment == 0:
			truth.append(2)
		else:
			truth.append(3)
	for score in output:
		if score > 0:
			predict.append(1)
		elif score == 0:
			predict.append(2)
		else:
			predict.append(3)

	f1_micro = f1_score(truth, predict, average='micro')
	f1_macro = f1_score(truth, predict, average='macro')

	right = 0
	for i in range(len(testing_data)):
		if truth[i] == predict[i]:
			right += 1

	with open('experiment.txt', 'a') as f:
		f.write('hidden {}, neuron {}\n'.format(FLAGS.rnn_units, FLAGS.hidden_layer))
		f.write('MSE: {}\n'.format(loss))
		f.write('F1_micro: {}, F1_macro: {}\n'.format(f1_micro, f1_macro))
		f.write('Acc: {}\n'.format(right/len(testing_data)))


if __name__ == '__main__':
	main()