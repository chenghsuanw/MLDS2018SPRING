import tensorflow as tf
import numpy as np
from data import Data
from tensorflow.python.layers import core as layers_core
from tqdm import tqdm

class CaptionGeneratorBasic(object):
	def __init__(self, FLAGS, w2v):
		self.FLAGS = FLAGS

		self.encoder_input = tf.placeholder(tf.float32, shape=[None, self.FLAGS.encoder_length, self.FLAGS.frame_dim], name='encoder_input')
		self.decoder_input_index = tf.placeholder(tf.int32, shape=[None, self.FLAGS.decoder_length], name='decoder_input_index')
		self.decoder_output_index = tf.placeholder(tf.int32, shape=[None, self.FLAGS.decoder_length], name='decoder_output_index')
		self.decoder_lengths = tf.placeholder(tf.int32, shape=[None], name="decoder_lengths")
		self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

		with tf.variable_scope('Embedding'):
			# w2v shape: [vocab_size, word_embed_dim]
			self.w2v = tf.get_variable('w2v', initializer=w2v)
			# decoder_input shape: [?, decoder_length, word_embed_dim]
			self.decoder_input = tf.nn.embedding_lookup(self.w2v, self.decoder_input_index)

		with tf.variable_scope('Encoder'):
			encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.FLAGS.rnn_units)
			self.encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob=self.keep_prob)
			# encoder_outputs shape: [?, encoder length, rnn units]
			# encoder_state: (c, h), each shape is [?, rnn_units]
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(self.encoder_cell, self.encoder_input, dtype=tf.float32)
			
		with tf.variable_scope('Decoder'):
			decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.FLAGS.rnn_units)
			self.decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=self.keep_prob)
			self.helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input, self.decoder_lengths)
			self.projection_layer = layers_core.Dense(self.w2v.shape[0], use_bias=False)
			self.initial_state = encoder_state
			
			self.decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.helper, self.initial_state, output_layer=self.projection_layer)

			final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder)
			logits = final_outputs.rnn_output

		with tf.name_scope('Loss'):
			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_output_index, logits=logits)
	
		with tf.name_scope('Optimize'):
			# Calculate and clip gradients
			params = tf.trainable_variables()
			gradients = tf.gradients(self.loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient)

			self.optimizer = tf.train.RMSPropOptimizer(self.FLAGS.lr_rate)
			self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, params))

	def train(self, data):
		with tf.Session() as sess:
	   		sess.run(tf.global_variables_initializer())
	   		saver = tf.train.Saver(max_to_keep=20)

	   		train_batches = len(data.encoder_input)//self.FLAGS.batch_size
	   		val_batches = len(data.val_encoder_input)//self.FLAGS.batch_size

	   		earlystop_chance = self.FLAGS.early_stop
	   		pre_val_loss = float('inf')

	   		for e in range(self.FLAGS.epochs):
	   			train_total_loss, val_total_loss = 0, 0

	   			# compute training loss
	   			for b in tqdm(range(train_batches)):
	   				encoder_input, decoder_input, decoder_output = data.next_batch(self.FLAGS.batch_size)
	   				feed_dict = {
	   					self.encoder_input: encoder_input,
	   					self.decoder_input_index: decoder_input,
	   					self.decoder_output_index: decoder_output,
	   					self.decoder_lengths: np.ones((encoder_input.shape[0]), dtype=int) * self.FLAGS.decoder_length,
	   					self.keep_prob: self.FLAGS.keep_prob
	   				}

	   				loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
	   				train_total_loss += np.mean(loss)/train_batches

	   			# compute validation loss
	   			for b in tqdm(range(val_batches)):
		   			val_encoder_input, val_decoder_input, val_decoder_output = data.next_batch(self.FLAGS.batch_size, val=True)
		   			feed_dict = {
		   				self.encoder_input: val_encoder_input,
		   				self.decoder_input_index: val_decoder_input,
		   				self.decoder_output_index: val_decoder_output,
		   				self.decoder_lengths: np.ones((val_encoder_input.shape[0]), dtype=int) * self.FLAGS.decoder_length,
		   				self.keep_prob: self.FLAGS.keep_prob
		   			}

		   			val_loss = sess.run(self.loss, feed_dict=feed_dict)
		   			val_total_loss += np.mean(val_loss)/val_batches
	   			
	   			print('epoch {}. loss: {}, val_loss: {}'.format(e, train_total_loss, val_total_loss))

	   			# validation loss improve
	   			if val_total_loss < pre_val_loss:
	   				pre_val_loss = val_total_loss
	   				earlystop_chance = self.FLAGS.early_stop
	   				path = saver.save(sess, self.FLAGS.model_path)
	   				print('save path: {}'.format(path))
	   			# validation loss not improve
	   			else:
	   				print('No improvement')
	   				earlystop_chance -= 1
	   				if not earlystop_chance:
	   					print('Early stop')
	   					break
   					
	def inference(self, data, model='./model/basic_model'):
   		with tf.Session() as sess:
	   		saver = tf.train.Saver()
	   		saver.restore(sess, model)
	   		print('load {}'.format(model))

	   		bos_id, eos_id = 1, 2
	   		inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.w2v, 
	   			tf.fill([self.FLAGS.batch_size], bos_id), eos_id)

	   		inference_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, inference_helper, 
	   			self.initial_state, output_layer=self.projection_layer)

	   		source_sequence_length = self.FLAGS.decoder_length
	   		maximum_iterations = tf.round(tf.reduce_max(source_sequence_length))

	   		final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
	   			inference_decoder, maximum_iterations=maximum_iterations)
	   		translations = final_outputs.sample_id

	   		feed_dict = {
	   			self.encoder_input: data,
	   			self.keep_prob: 1
	   		}

	   		captions = sess.run([translations], feed_dict=feed_dict)
	   		
	   	return captions[0]

class CaptionGeneratorAttention(object):
	def __init__(self, FLAGS, w2v):
		self.FLAGS = FLAGS

		self.encoder_input = tf.placeholder(tf.float32, shape=[None, self.FLAGS.encoder_length, self.FLAGS.frame_dim], name='encoder_input')
		self.decoder_input_index = tf.placeholder(tf.int32, shape=[None, self.FLAGS.decoder_length], name='decoder_input_index')
		self.decoder_output_index = tf.placeholder(tf.int32, shape=[None, self.FLAGS.decoder_length], name='decoder_output_index')
		self.decoder_lengths = tf.placeholder(tf.int32, shape=[None], name="decoder_lengths")
		self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

		with tf.variable_scope('Embedding'):
			# w2v shape: [vocab_size, word_embed_dim]
			self.w2v = tf.get_variable('w2v', initializer=w2v)
			# decoder_input shape: [?, decoder_length, word_embed_dim]
			self.decoder_input = tf.nn.embedding_lookup(self.w2v, self.decoder_input_index)

		with tf.variable_scope('Encoder'):
			encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.FLAGS.rnn_units)
			self.encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob=self.FLAGS.keep_prob)
			# encoder_outputs shape: [?, encoder length, rnn units]
			# encoder_state: (c, h), each shape is [?, rnn_units]
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(self.encoder_cell, self.encoder_input, dtype=tf.float32)

		with tf.variable_scope('Decoder'):
			decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.FLAGS.rnn_units)
			self.decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=self.FLAGS.keep_prob)
			self.helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input, self.decoder_lengths)
			self.projection_layer = layers_core.Dense(self.w2v.shape[0], use_bias=False)
			
			attention_states = encoder_outputs
			attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.FLAGS.rnn_units, attention_states)
			self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism, attention_layer_size=self.FLAGS.rnn_units)
			self.initial_state = self.decoder_cell.zero_state(tf.shape(self.encoder_input)[0], tf.float32).clone(cell_state=encoder_state)
			self.decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.helper, self.initial_state, output_layer=self.projection_layer)

			# final_outputs: (rnn_output, sample_id), rnn_output shape: [?, decoder_length, vocab_size], sample_id shape: [?, decoder_length]
			# final_state: AttentionWrapperState
			# final_sequence_lengths shape: [?]
			final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder)
			logits = final_outputs.rnn_output

		with tf.name_scope('Loss'):
			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_output_index, logits=logits)
	
		with tf.name_scope('Optimize'):
			# Calculate and clip gradients
			params = tf.trainable_variables()
			gradients = tf.gradients(self.loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient)

			self.optimizer = tf.train.RMSPropOptimizer(self.FLAGS.lr_rate)
			self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, params))

	def train(self, data):
   		with tf.Session() as sess:
	   		sess.run(tf.global_variables_initializer())
	   		saver = tf.train.Saver(max_to_keep=20)

	   		train_batches = len(data.encoder_input)//self.FLAGS.batch_size
	   		val_batches = len(data.val_encoder_input)//self.FLAGS.batch_size

	   		earlystop_chance = self.FLAGS.early_stop
	   		pre_val_loss = float('inf')

	   		for e in range(self.FLAGS.epochs):
	   			train_total_loss, val_total_loss = 0, 0

	   			# compute training loss
	   			for b in tqdm(range(train_batches)):
	   				encoder_input, decoder_input, decoder_output = data.next_batch(self.FLAGS.batch_size)
	   				feed_dict = {
	   					self.encoder_input: encoder_input,
	   					self.decoder_input_index: decoder_input,
	   					self.decoder_output_index: decoder_output,
	   					self.decoder_lengths: np.ones((encoder_input.shape[0]), dtype=int) * self.FLAGS.decoder_length,
	   					self.keep_prob: self.FLAGS.keep_prob
	   				}

	   				loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
	   				train_total_loss += np.mean(loss)/train_batches

	   			# compute validation loss
	   			for b in tqdm(range(val_batches)):
		   			val_encoder_input, val_decoder_input, val_decoder_output = data.next_batch(self.FLAGS.batch_size, val=True)
		   			feed_dict = {
		   				self.encoder_input: val_encoder_input,
		   				self.decoder_input_index: val_decoder_input,
		   				self.decoder_output_index: val_decoder_output,
		   				self.decoder_lengths: np.ones((val_encoder_input.shape[0]), dtype=int) * self.FLAGS.decoder_length,
		   				self.keep_prob: self.FLAGS.keep_prob
		   			}

		   			val_loss = sess.run(self.loss, feed_dict=feed_dict)
		   			val_total_loss += np.mean(val_loss)/val_batches
	   			
	   			print('epoch {}. loss: {}, val_loss: {}'.format(e, train_total_loss, val_total_loss))

	   			# validation loss improve
	   			if val_total_loss < pre_val_loss:
	   				pre_val_loss = val_total_loss
	   				earlystop_chance = self.FLAGS.early_stop
	   				path = saver.save(sess, self.FLAGS.model_path)
	   				print('save path: {}'.format(path))
	   			# validation loss not improve
	   			else:
	   				print('No improvement')
	   				earlystop_chance -= 1
	   				if not earlystop_chance:
	   					print('Early stop')
	   					break

	def inference(self, data, model='./model/attention_model'):
   		with tf.Session() as sess:
	   		
	   		bos_id, eos_id = 1, 2
	   		inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.w2v, tf.fill([self.FLAGS.batch_size], bos_id), eos_id)

	   		inference_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, inference_helper, self.initial_state, output_layer=self.projection_layer)

	   		source_sequence_length = self.FLAGS.decoder_length
	   		maximum_iterations = tf.round(tf.reduce_max(source_sequence_length))

	   		final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(inference_decoder, maximum_iterations=maximum_iterations)
	   		translations = final_outputs.sample_id

	   		saver = tf.train.Saver()
	   		saver.restore(sess, model)
	   		print('load {}'.format(model))

	   		feed_dict = {
	   			self.encoder_input: data,
	   			self.keep_prob: 1
	   		}

	   		captions = sess.run([translations], feed_dict=feed_dict)
   		
   		return captions[0]






