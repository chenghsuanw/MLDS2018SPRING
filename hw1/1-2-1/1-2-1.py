import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import pickle
from sklearn.decomposition import PCA


def fit_function(x):
	return pow(x,3)

def count_parameters():
	total = 0
	for variable in tf.trainable_variables():
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total += variable_parameters
	print('total trainable parameters', total)
	return total

def main():
	
	with tf.name_scope('Input'):
		x = tf.placeholder('float', [None, 1], name='x')

	with tf.variable_scope('Hidden_Layers'):
		W1 = tf.get_variable('W1', initializer=tf.random_normal([1, 10], stddev=0.1))
		b1 = tf.get_variable('b1', initializer=tf.random_normal([10], stddev=0.1))
		layer1 = tf.nn.leaky_relu(tf.matmul(x, W1) + b1)

		W2 = tf.get_variable('W2', initializer=tf.random_normal([10, 18], stddev=0.1))
		b2 = tf.get_variable('b2', initializer=tf.random_normal([18], stddev=0.1))
		layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)

		W3 = tf.get_variable('W3', initializer=tf.random_normal([18, 15], stddev=0.1))
		b3 = tf.get_variable('b3', initializer=tf.random_normal([15], stddev=0.1))
		layer3 = tf.nn.leaky_relu(tf.matmul(layer2, W3) + b3)

		W4 = tf.get_variable('W4', initializer=tf.random_normal([15, 10], stddev=0.1))
		b4 = tf.get_variable('b4', initializer=tf.random_normal([10], stddev=0.1))
		layer4 = tf.nn.leaky_relu(tf.matmul(layer3, W4) + b4)

		W5 = tf.get_variable('W5', initializer=tf.random_normal([10, 10], stddev=0.1))
		b5 = tf.get_variable('b5', initializer=tf.random_normal([10], stddev=0.1))
		layer5 = tf.nn.leaky_relu(tf.matmul(layer4, W5) + b5)

		W6 = tf.get_variable('W6', initializer=tf.random_normal([10, 4], stddev=0.1))
		b6 = tf.get_variable('b6', initializer=tf.random_normal([4], stddev=0.1))
		layer6 = tf.nn.leaky_relu(tf.matmul(layer5, W6) + b6)

		W7 = tf.get_variable('W7', initializer=tf.random_normal([4, 1], stddev=0.1))
		b7 = tf.get_variable('b7', initializer=tf.random_normal([1], stddev=0.1))
		predict = tf.matmul(layer6, W7) + b7

	with tf.name_scope('Loss'):
		label = tf.placeholder('float', [None, 1], name='label')
		loss = tf.reduce_sum(tf.losses.mean_squared_error(predict, label), name='Loss')

	with tf.name_scope('Optimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate=10e-4).minimize(loss)


	x_train = (np.array(range(1, 10000))/10000).reshape(-1, 1)
	y_train = np.array(list(map(fit_function, x_train))).reshape(-1, 1)

	total_epochs = 500
	batch_size = 128
	total_batches = int(x_train.shape[0]/batch_size)

	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.allow_growth = True

	# p = count_parameters()
	weights_layer1, weights_total, los_list = [], [], []
	for train_event in range(8):
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(total_epochs):
				for batch in range(total_batches):
					batch_x = x_train[batch*batch_size: (batch+1)*batch_size].reshape(-1, 1)
					batch_y = y_train[batch*batch_size: (batch+1)*batch_size].reshape(-1, 1)
					sess.run(optimizer, feed_dict={x: batch_x, label: batch_y})

				los = sess.run(loss, feed_dict={x: x_train, label: y_train})
				print('epoch: {} loss: {}'.format(epoch+1, los))

				# record the weights
				if epoch % 3 == 0:
					with tf.variable_scope('Hidden_Layers', reuse=True):
						w1 = tf.get_variable(name='W1', shape=[1, 10]).eval().reshape(-1,)
						w2 = tf.get_variable(name='W2', shape=[10, 18]).eval().reshape(-1,)
						w3 = tf.get_variable(name='W3', shape=[18, 15]).eval().reshape(-1,)
						w4 = tf.get_variable(name='W4', shape=[15, 10]).eval().reshape(-1,)
						w5 = tf.get_variable(name='W5', shape=[10, 10]).eval().reshape(-1,)
						w6 = tf.get_variable(name='W6', shape=[10, 4]).eval().reshape(-1,)
						w7 = tf.get_variable(name='W7', shape=[4, 1]).eval().reshape(-1,)
						weights_layer1.append(w1)
						weights_total.append(np.concatenate((w1, w2, w3, w4, w5, w6, w7)))
						los_list.append(los)

	loss = np.array(los_list)
	weights_total = np.array(weights_total)
	weights_layer1 = np.array(weights_layer1)

	pcat = PCA(n_components=2, whiten=True)
	pca1 = PCA(n_components=2)
	wt = pcat.fit_transform(weights_total)
	w1 = pca1.fit_transform(weights_layer1)
	
	# with open('weights_total.pickle', 'wb') as f:
	# 	pickle.dump(wt, f)
	# with open('weights_layer1.pickle', 'wb') as f:
	# 	pickle.dump(w1, f)
	# with open('loss.pickle', 'wb') as f:
	# 	pickle.dump(loss, f)
	
	plt.axis([min(wt[:,0])-1, max(wt[:,0])+1, min(wt[:,1])-1, max(wt[:,1])+1])
	plt.title('whole model')
	for j in range(loss.shape[0]):
		plt.text(wt[j][0], wt[j][1], '%.2f' % loss[j])
	plt.savefig('visualize_total.png')
	
	plt.clf()
	plt.axis([min(w1[:,0])-1, max(w1[:,0])+1, min(w1[:,1])-1, max(w1[:,1])+1])
	plt.title('first layer')
	for j in range(loss.shape[0]):
		plt.text(w1[j][0], w1[j][1], '%.2f' % loss[j])
	plt.savefig('visualize_layer1.png')


if __name__ == '__main__':
	main()