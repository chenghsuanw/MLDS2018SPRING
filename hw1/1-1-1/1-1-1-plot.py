import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


x = np.linspace(-1, 1, 1000)
y = np.sin(x * np.pi * 5) / (5 * np.pi *x)
# y = (1/np.sqrt(5)) * (((1 + np.sqrt(5)) / 2) ** x - (2 / (1 + np.sqrt(5))) ** x * np.cos(x*np.pi))
x, y = torch.unsqueeze(torch.from_numpy(x), dim=1), torch.unsqueeze(torch.from_numpy(y), dim=1)
x, y = Variable(x).float(), Variable(y).float()

net_1 = torch.load('./models/fun2_shallow.pkl')
net_2 = torch.load('./models/fun2_middle.pkl')
net_3 = torch.load('./models/fun2_deep.pkl')

pred_1 = net_1(x)
pred_2 = net_2(x)
pred_3 = net_3(x)

loss_1 = pickle.load(open('./loss/loss_1.pkl', 'rb'))
loss_2 = pickle.load(open('./loss/loss_2.pkl', 'rb'))
loss_3 = pickle.load(open('./loss/loss_3.pkl', 'rb'))


plt.cla()
plt.plot(x.data.numpy(), y.data.numpy(), 'b-', lw=1)
plt.plot(x.data.numpy(), pred_1.data.numpy(), 'y-', lw=1, label='shallow')
plt.plot(x.data.numpy(), pred_2.data.numpy(), 'r-', lw=1, label='middle')
plt.plot(x.data.numpy(), pred_3.data.numpy(), 'g-', lw=1, label='deep')
plt.legend(loc='upper right')
# plt.text(5, -10, 'Loss=%.4f' % loss.data[0])
plt.savefig('hw1_1_fit2.png')


plt.cla()
plt.plot(loss_1, label='shallow')
plt.plot(loss_2, label='middle')
plt.plot(loss_3, label='deep')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.savefig('hw1_1_loss2.png')
