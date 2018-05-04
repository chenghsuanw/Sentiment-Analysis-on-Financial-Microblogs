import numpy as np
import tensorflow as tf
from tqdm import tqdm

class LSTM(object):
	def __init__(self, FLAGS, w2v):
		self.sess = tf.Session()
		self.FLAGS = FLAGS

		self.input_seq_index = tf.placeholder(tf.int32, shape=[None, FLAGS.max_length], name='input_seq_index')
		self.output_score = tf.placeholder(tf.float32, shape=[None, 1], name='output_score')

		with tf.variable_scope('Embedding'):
			self.w2v = tf.get_variable('w2v', initializer=w2v)
			self.input_embed = tf.nn.embedding_lookup(self.w2v, self.input_seq_index)

		with tf.variable_scope('LSTM'):
			self.cell = tf.contrib.rnn.BasicLSTMCell(self.FLAGS.rnn_units)
			self.inital_state = self.cell.zero_state(tf.shape(self.input_embed)[0], tf.float32)

			state = self.inital_state
			for time_step in range(self.FLAGS.max_length):
				# get_variable_scope(): returns the current variable scope
				# reuse_variables(): reuse variables in this scope
				# tf.get_variable_scope().reuse_variables()
				# cell(): run this RNN cell on inputs, starting from the given state.
				cell_output, state = self.cell(self.input_embed[:, time_step, :], state)

			# final state: (c, h), c is the hidden state and h is the output
			# shape: [None, rnn_units]
			self.final_state = state

		with tf.variable_scope('Fully_Connect'):
			W1 = tf.get_variable('W1', initializer=tf.random_normal([self.FLAGS.rnn_units, self.FLAGS.hidden_layer], stddev=0.1))
			b1 = tf.get_variable('b1', initializer=tf.random_normal([self.FLAGS.hidden_layer], stddev=0.1))
			layer1 = tf.nn.leaky_relu(tf.matmul(self.final_state[0], W1) + b1)

			W2 = tf.get_variable('W2', initializer=tf.random_normal([self.FLAGS.hidden_layer, 1], stddev=0.1))
			b2 = tf.get_variable('b2', initializer=tf.random_normal([1], stddev=0.1))
			self.output = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)
		
		with tf.name_scope('Loss'):
			self.loss = tf.reduce_sum(tf.losses.mean_squared_error(self.output, self.output_score))

		with tf.name_scope('Optimizer'):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=10e-5)
			self.train_op = self.optimizer.minimize(self.loss)


	def train(self, data, label, save_path='./model/model.ckpt'):
		self.sess.run(tf.global_variables_initializer())

		total_batch = data.shape[0] // self.FLAGS.batch_size

		for e in range(self.FLAGS.epochs):
			los = 0
			for b in tqdm(range(total_batch)):
				feed_dict = {
				self.input_seq_index: data[b*self.FLAGS.batch_size: (b+1)*self.FLAGS.batch_size],
				self.output_score: label[b*self.FLAGS.batch_size: (b+1)*self.FLAGS.batch_size].reshape([-1, 1])
				}
				l, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
				los += l/self.FLAGS.batch_size

			print('epoch {}, loss: {}'.format(e, los))


		saver = tf.train.Saver()
		saver.save(self.sess, save_path)


	def test(self, data, label, model_path='./model/model.ckpt'):
		saver = tf.train.Saver()
		saver.restore(self.sess, model_path)
		print('Load model')
		
		feed_dict = {
		self.input_seq_index: data,
		self.output_score: label.reshape([-1, 1])
		}
		loss, output = self.sess.run([self.loss, self.output], feed_dict=feed_dict)
		
		print('testing error: {}'.format(loss))

		return output, loss
