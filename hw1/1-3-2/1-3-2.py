import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import pickle

def count_parameters():
	total = 0
	for variable in tf.trainable_variables():
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total += variable_parameters

	return total

def train(para1, para2, para3):
	mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

	tf.reset_default_graph()
	with tf.name_scope('Input'):
		x = tf.placeholder('float', shape=[None, 784], name='x')

	with tf.variable_scope('Hidden_Layers'):
			W1 = tf.get_variable('W1', initializer=tf.random_normal([784, para1], stddev=0.1))
			b1 = tf.get_variable('b1', initializer=tf.random_normal([para1], stddev=0.1))
			layer1 = tf.nn.leaky_relu(tf.matmul(x, W1) + b1)

			W2 = tf.get_variable('W2', initializer=tf.random_normal([para1, para2], stddev=0.1))
			b2 = tf.get_variable('b2', initializer=tf.random_normal([para2], stddev=0.1))
			layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)

			W3 = tf.get_variable('W3', initializer=tf.random_normal([para2, para3], stddev=0.1))
			b3 = tf.get_variable('b3', initializer=tf.random_normal([para3], stddev=0.1))
			layer3 = tf.nn.leaky_relu(tf.matmul(layer2, W3) + b3)

			W4 = tf.get_variable('W4', initializer=tf.random_normal([para3, 10], stddev=0.1))
			b4 = tf.get_variable('b4', initializer=tf.random_normal([10], stddev=0.1))
			predict = tf.nn.leaky_relu(tf.matmul(layer3, W4) + b4)
			# predict = tf.nn.softmax(layer4)

	with tf.name_scope('optimizer'):
		label = tf.placeholder('float', shape=[None, 10], name='label')
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=label))
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	with tf.name_scope('evaluate'):
		correct = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	p_count = count_parameters()
	trainEpochs = 50
	batchSize = 128
	totalBatchs = int(mnist.train.num_examples/batchSize)
	
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.allow_growth = True

	test_x = mnist.test.images
	test_y = mnist.test.labels
	train_x = mnist.train.images
	train_y = mnist.train.labels

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(trainEpochs):
			for i in range(totalBatchs):
				sess.run(optimizer, feed_dict={x: train_x[i*batchSize: (i+1)*batchSize], label: train_y[i*batchSize: (i+1)*batchSize]})

			los_train, acc_train = sess.run([loss, accuracy], feed_dict={x: train_x[:10000], label: train_y[:10000]})
			print('epoch: {} loss: {} acc: {}'.format(epoch+1, los_train, acc_train))

		los_train, acc_train = sess.run([loss, accuracy], feed_dict={x: train_x, label: train_y})
		los_test, acc_test = sess.run([loss, accuracy], feed_dict={x: test_x, label: test_y})

	return p_count, los_train, acc_train, los_test, acc_test

def main():
	
	layer1 = [2, 4, 8, 16]
	layer2 = [4, 8, 16, 32]
	layer3 = [8, 16, 32, 64]

	para_list = []
	acc_train_list, loss_train_list = [], []
	acc_test_list, loss_test_list = [], []

	for p1 in layer1:
		for p2 in layer2:
			for p3 in layer3:
				p_count, los_train, acc_train, los_test, acc_test = train(p1, p2, p3)
				para_list.append(p_count)
				acc_train_list.append(acc_train)
				acc_test_list.append(acc_test)
				loss_train_list.append(los_train)
				loss_test_list.append(los_test)

	plt.scatter(para_list, loss_train_list)
	plt.scatter(para_list, loss_test_list)
	plt.legend(['train_loss', 'test_loss'])
	plt.savefig('loss.png')
	plt.clf()
	plt.scatter(para_list, acc_train_list)
	plt.scatter(para_list, acc_test_list)
	plt.legend(['train_acc', 'test_acc'])
	plt.savefig('acc.png')

if __name__ == '__main__':
	main()