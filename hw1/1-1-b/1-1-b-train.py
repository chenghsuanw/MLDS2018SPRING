import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import pickle
import matplotlib.pyplot as plt

EPOCH = 1000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CNN_shallow(nn.Module):
    def __init__(self):
        super(CNN_shallow, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out1 = nn.Linear(32*16*16, 21)
        self.out2 = nn.Linear(21, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out1(x)
        output = F.relu(output)
        output = self.out2(output)
        return output


class CNN_middle(nn.Module):
    def __init__(self):
        super(CNN_middle, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 45, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out1 = nn.Linear(45*8*8, 50)
        self.out2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out1(x)
        output = F.relu(output)
        output = self.out2(output)
        return output


class CNN_deep(nn.Module):
    def __init__(self):
        super(CNN_deep, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 105, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(105, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out1 = nn.Linear(64*2*2, 50)
        self.out2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.out1(x)
        output = F.relu(output)
        output = self.out2(output)
        return output


def main():

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=2)

	net = CNN_shallow().cuda()

	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	loss_func = nn.CrossEntropyLoss()

	loss_ls = []
	acc_ls = []

	for epoch in range(EPOCH):
		for step, (x, y) in enumerate(trainloader):
			b_x = Variable(x.cuda())
			b_y = Variable(y.cuda())

			output = net(b_x)
			loss = loss_func(output, b_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		pred = torch.max(output, 1)[1].data.squeeze()
		acc = sum(pred == b_y.data) / float(b_y.data.size(0))
		print('epoch: {}  | step: {} \t| loss: {} | acc: {}'.format(epoch, step, loss.data[0], acc))
		loss_ls.append(loss.data[0])
		acc_ls.append(acc)
		pickle.dump(loss_ls, open('./cabin/cnn_shallow_loss.pk', 'wb'))
		pickle.dump(acc_ls, open('./cabin/cnn_shallow_acc.pk', 'wb'))

	torch.save(net, './cabin/cnn_shallow_model.pkl')


if __name__ == '__main__':
    main()