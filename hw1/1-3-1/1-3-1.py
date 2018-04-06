import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import pickle

def main():
	mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

	with tf.name_scope('Input'):
		x = tf.placeholder('float', shape=[None, 784], name='x')

	with tf.variable_scope('Hidden_Layers'):
		W1 = tf.get_variable('W1', initializer=tf.random_normal([784, 256], stddev=0.1))
		b1 = tf.get_variable('b1', initializer=tf.random_normal([256], stddev=0.1))
		layer1 = tf.nn.leaky_relu(tf.matmul(x, W1) + b1)

		W2 = tf.get_variable('W2', initializer=tf.random_normal([256, 256], stddev=0.1))
		b2 = tf.get_variable('b2', initializer=tf.random_normal([256], stddev=0.1))
		layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)

		W3 = tf.get_variable('W3', initializer=tf.random_normal([256, 256], stddev=0.1))
		b3 = tf.get_variable('b3', initializer=tf.random_normal([256], stddev=0.1))
		layer3 = tf.nn.leaky_relu(tf.matmul(layer2, W3) + b3)

		W4 = tf.get_variable('W4', initializer=tf.random_normal([256, 10], stddev=0.1))
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

	trainEpochs = 1500
	batchSize = 128
	totalBatchs = int(mnist.train.num_examples/batchSize)
	epoch_list = []
	accuracy_train_list, loss_train_list = [], []
	accuracy_test_list, loss_test_list = [], []

	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.allow_growth = True

	test_x = mnist.test.images
	test_y = mnist.test.labels
	train_x = mnist.train.images
	train_y = mnist.train.labels
	shuffle_y = np.random.permutation(train_y)

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(trainEpochs):
			for i in range(totalBatchs):
				sess.run(optimizer, feed_dict={x: train_x[i*batchSize: (i+1)*batchSize], label: shuffle_y[i*batchSize: (i+1)*batchSize]})

			los_train, acc_train = sess.run([loss, accuracy], feed_dict={x: train_x[:10000], label: shuffle_y[:10000]})
			los_test, acc_test = sess.run([loss, accuracy], feed_dict={x: test_x, label: test_y})
			epoch_list.append(epoch)
			loss_train_list.append(los_train)
			accuracy_train_list.append(acc_train)
			loss_test_list.append(los_test)
			accuracy_test_list.append(los_test)
			print('epoch: {} loss: {} acc: {}'.format(epoch+1, los_train, acc_train))

	plt.plot(epoch_list, loss_train_list)
	plt.plot(epoch_list, loss_test_list)
	plt.legend(['train', 'test'])
	plt.savefig('fit_random_label.png')


if __name__ == '__main__':
	main()