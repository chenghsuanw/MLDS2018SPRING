import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import pickle
import matplotlib.pyplot as plt

EPOCH = 20000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


net_1 = torch.nn.Sequential(
    torch.nn.Linear(1, 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 1)
)

net_2 = torch.nn.Sequential(
    torch.nn.Linear(1, 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 15),
    torch.nn.ReLU(),
    torch.nn.Linear(15, 1)
)

net_3 = torch.nn.Sequential(
    torch.nn.Linear(1, 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)



def main():

	x = np.linspace(-1, 1, 1000)

	# we fit 2 functions as below

	# func 1
	y = np.sin(x * np.pi * 5) / (5 * np.pi *x)

	# func 2
	# x = np.linspace(-5, 5, 1000)
	# y = (1/np.sqrt(5)) * (((1 + np.sqrt(5)) / 2) ** x - (2 / (1 + np.sqrt(5))) ** x * np.cos(x*np.pi))

	x, y = torch.unsqueeze(torch.from_numpy(x), dim=1), torch.unsqueeze(torch.from_numpy(y), dim=1)
	x, y = Variable(x).float(), Variable(y).float()

	for training

	loss_ls = []
	net = net_1

	optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
	loss_func = torch.nn.MSELoss()

	for t in range(EPOCH):
	    prediction = net(x)
	    loss = loss_func(prediction, y)
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()
	    loss_ls.append(loss.data[0])

	pickle.dump(loss_ls, open('./loss/loss_1.pkl', 'wb'))
	pickle.dump(net, open('./models/shallow.pkl', 'wb'))


if __name__ == '__main__':
    main()