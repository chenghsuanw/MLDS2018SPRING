import numpy as np
import pickle
import json
import re
from collections import Counter


special_tokens = ['<UNK>', '<BOS>', '<EOS>', '<PAD>']

def load_training_data(path):
	print('Loading training data...')

	ids, captions = [], []
	labels = json.load(open(path))
	for label in labels:
		ids.append(label['id'])
		captions.append(label['caption'])

	train_feature = []
	train_label = []
	for i, ID in enumerate(ids):
		feat = np.load('./MLDS_hw2_1_data/training_data/feat/'+ID+'.npy')
		for label in captions[i]:
			train_feature.append(feat)
			train_label.append(re.sub(r'[^\w\s]', '', label.lower()))
	
	train_data = np.array(list(zip(train_feature, train_label)))
	# shuffle the data
	# train_data = np.random.permutation(train_data)

	print('There are {} training data pair (feature, caption)'.format(train_data.shape[0]))

	# shape: [None, (feature, label)]
	return train_data

def load_testing_data(path):
	print('Loading testing data...')

	with open(path+'/id.txt') as f:
		IDs = f.readlines()
	features = []
	ids = []
	for ID in IDs:
		ID = ID.strip()
		feat = np.load(path+'/feat/'+ID+'.npy')
		ids.append(ID)
		features.append(feat)

	features = np.array(features)

	print('There are {} testing data with shape ({}, {})'.format(features.shape[0], features.shape[1], features.shape[2]))

	return ids, features

def build_dictionary(sentences, min_count=1):
	print('Building dictionary...')
	
	w2i = dict()
	index = 0

	for word in special_tokens:
		w2i[word] = index
		index += 1

	cnt = Counter()
	for sentence in sentences:
		cnt.update(sentence.split())

	for word, count in cnt.items():
		if count >= min_count:
			w2i[word] = index
			index += 1

	with open('w2i.pickle', 'wb') as f:
		pickle.dump(w2i, f)

	print('There are {} tokens in the dictionary.'.format(len(w2i)))

def build_word_vector(w2i, w2v_path, word_embed_dim):
	print('Building word vectors...')

	w2v_dict = dict()
	with open(w2v_path) as f:
		content = f.readlines()

	for line in content:
		word, vec = line.strip().split(' ', 1)
		w2v_dict[word] = np.loadtxt([vec], dtype=np.float32)

	w2v = []
	for word in w2i.keys():
		# random assign word embedding for special tokens
		if word in special_tokens:
			w2v.append(np.random.normal(0, 0.1, word_embed_dim).astype(np.float32))
		# assign pretrained word embedding
		elif word in w2v_dict:
			w2v.append(w2v_dict[word])
		# assign <UNK> embedding for the word not in the pretrained dictionary
		else:
			w2v.append(w2v[0])

	w2v = np.array(w2v)
	with open('w2v.pickle', 'wb') as f:
		pickle.dump(w2v, f)

	print('There are {} word vectors. Each vector has {} dimension.'.format(len(w2v), word_embed_dim))







