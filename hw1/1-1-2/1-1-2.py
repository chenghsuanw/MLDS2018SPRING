import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pickle

def main():

	epoch_list = np.array(range(1, 101))
	with open('loss_DNN3.pickle', 'rb') as f:
		loss_shallow = pickle.load(f)
	with open('loss_DNN4.pickle', 'rb') as f:
		loss_middle = pickle.load(f)
	with open('loss_DNN8.pickle', 'rb') as f:
		loss_deep = pickle.load(f)
	with open('acc_DNN3.pickle', 'rb') as f:
		acc_shallow = pickle.load(f)
	with open('acc_DNN4.pickle', 'rb') as f:
		acc_middle = pickle.load(f)
	with open('acc_DNN8.pickle', 'rb') as f:
		acc_deep = pickle.load(f)

	plt.plot(epoch_list, loss_deep)
	plt.plot(epoch_list, loss_middle)
	plt.plot(epoch_list, loss_shallow)
	plt.legend(['deep' ,'middle', 'shallow'])
	plt.savefig('loss.png')

	plt.clf()

	plt.plot(epoch_list, acc_deep)
	plt.plot(epoch_list, acc_middle)
	plt.plot(epoch_list, acc_shallow)
	plt.legend(['deep' ,'middle', 'shallow'])
	plt.savefig('acc.png')

if __name__ == '__main__':
	main()