import numpy as np
import tensorflow as tf
import pickle
import json
import os
from collections import Counter
from model import *
from data import *
from preprocess import *
import csv
import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_label', default='./MLDS_hw2_1_data/training_label.json', help='training label path')
tf.app.flags.DEFINE_string('test_file', default='./MLDS_hw2_1_data/testing_data', help='testing data file path')
tf.app.flags.DEFINE_string('word2vec', default='./glove.txt', help='pretrained word2vector path')
tf.app.flags.DEFINE_string('output_file', default='./answer.csv', help='output file path')
tf.app.flags.DEFINE_string('model', default='basic', help='basic or attention')
tf.app.flags.DEFINE_string('model_path', default='./model.ckpt', help='model path')
tf.app.flags.DEFINE_integer('word_embed_dim', default=300, help='dimension of the word embedding')
tf.app.flags.DEFINE_integer('rnn_units', default=512, help='number of hidden units of rnn cell')
tf.app.flags.DEFINE_integer('encoder_length', default=80, help='number of timestamp of encoder(frame number)')
tf.app.flags.DEFINE_integer('frame_dim', default=4096, help='dimension of the frame')
tf.app.flags.DEFINE_integer('decoder_length', default=20, help='number of timestamp of decoder(caption length)')
tf.app.flags.DEFINE_integer('epochs', default=100, help='epochs when training')
tf.app.flags.DEFINE_integer('batch_size', default=100, help='batch size per iteration')
tf.app.flags.DEFINE_integer('early_stop', default=3, help='chance without improvement of validation loss')
tf.app.flags.DEFINE_float('max_gradient', default=5.0, help='max gradient when training')
tf.app.flags.DEFINE_float('lr_rate', default=0.001, help='learning rate')
tf.app.flags.DEFINE_float('keep_prob', default=0.9, help='keep ratio when dropout')

test_data_dir = sys.argv[1]
output_file = sys.argv[2]

def index_to_string(sentences, w2i):
	i2w = dict()
	for word, index in w2i.items():
		i2w[index] = word

	ans = []
	for sentence in sentences:
		s = ''
		for i in sentence:
			if i2w[i] == '<EOS>' or i2w[i] == '<PAD>':
				break
			elif i2w[i] == '<BOS>' or i2w[i] == '<UNK>':
				continue
			else:
				s += ' '+i2w[i]

		ans.append(s)

	return ans

def main():

	tf.reset_default_graph()
	
	with open('w2i.pickle', 'rb') as f:
		w2i = pickle.load(f)

	with open('w2v.pickle', 'rb') as f:
		w2v = pickle.load(f)

	model = CaptionGeneratorBasic(FLAGS, w2v)

	IDs, testing_data = load_testing_data(test_data_dir)

	predict_index = model.inference(testing_data, FLAGS.model_path)
	predict = index_to_string(predict_index, w2i)
	answer = list(zip(IDs, predict))

	with open(output_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for ans in answer:
			writer.writerow([ans[0], ans[1]])

if __name__ == '__main__':
	main()

