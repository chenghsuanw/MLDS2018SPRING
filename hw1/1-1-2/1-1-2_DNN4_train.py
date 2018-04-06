import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import math
import pickle

def main():
	#134794
	mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

	with tf.name_scope('Input'):
		x = tf.placeholder('float', [None, 784], name='x')

	with tf.variable_scope('Hidden_Layers'):
		W1 = tf.get_variable('W1', initializer=tf.random_normal([784, 128], stddev=0.1))
		b1 = tf.get_variable('b1', initializer=tf.random_normal([128], stddev=0.1))
		layer1 = tf.nn.leaky_relu(tf.matmul(x, W1) + b1)

		W2 = tf.get_variable('W2', initializer=tf.random_normal([128, 128], stddev=0.1))
		b2 = tf.get_variable('b2', initializer=tf.random_normal([128], stddev=0.1))
		layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)

		W3 = tf.get_variable('W3', initializer=tf.random_normal([128, 128], stddev=0.1))
		b3 = tf.get_variable('b3', initializer=tf.random_normal([128], stddev=0.1))
		layer3 = tf.nn.leaky_relu(tf.matmul(layer2, W3) + b3)

		W4 = tf.get_variable('W4', initializer=tf.random_normal([128, 10], stddev=0.1))
		b4 = tf.get_variable('b4', initializer=tf.random_normal([10], stddev=0.1))
		predict = tf.nn.leaky_relu(tf.matmul(layer3, W4) + b4)

	with tf.name_scope('Loss'):
		label = tf.placeholder('float', [None, 10], name='label')
		loss = tf.reduce_sum(tf.losses.mean_squared_error(predict, label))

	with tf.name_scope('Optimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate=10e-5).minimize(loss)

	with tf.name_scope('evaluate'):
		correct = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


	trainEpochs = 100
	batchSize = 64
	totalBatchs = int(mnist.train.num_examples/batchSize)
	epoch_list = []
	accuracy_train_list, loss_train_list = [], []

	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.allow_growth = True

	train_x = mnist.train.images
	train_y = mnist.train.labels

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(trainEpochs):
			for i in range(totalBatchs):
				sess.run(optimizer, feed_dict={x: train_x[i*batchSize: (i+1)*batchSize], label: train_y[i*batchSize: (i+1)*batchSize]})

			los_train, acc_train = sess.run([loss, accuracy], feed_dict={x: train_x[:10000], label: train_y[:10000]})
			epoch_list.append(epoch)
			loss_train_list.append(los_train)
			accuracy_train_list.append(acc_train)
			print('epoch: {} loss: {} acc: {}'.format(epoch+1, los_train, acc_train))

	with open('loss_DNN4.pickle', 'wb') as f:
		pickle.dump(loss_train_list, f)
	with open('acc_DNN4.pickle', 'wb') as f:
		pickle.dump(accuracy_train_list, f)


	

if __name__ == '__main__':
	main()