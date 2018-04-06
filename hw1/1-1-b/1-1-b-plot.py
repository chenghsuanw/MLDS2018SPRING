import pickle
import matplotlib.pyplot as plt


cnn_deep_loss = pickle.load(open('./cabin/cnn_deep_loss.pk', 'rb'))
cnn_middle_loss = pickle.load(open('./cabin/cnn_middle_loss.pk', 'rb'))
cnn_shallow_loss = pickle.load(open('./cabin/cnn_shallow_loss.pk', 'rb'))

cnn_deep_acc = pickle.load(open('./cabin/cnn_deep_acc.pk', 'rb'))
cnn_middle_acc = pickle.load(open('./cabin/cnn_middle_acc.pk', 'rb'))
cnn_shallow_acc = pickle.load(open('./cabin/cnn_shallow_acc.pk', 'rb'))

# plot loss curve
plt.plot(cnn_deep_loss, label='deep')
plt.plot(cnn_middle_loss, label='middle')
plt.plot(cnn_shallow_loss, label='shallow')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.savefig('./1_1_2_loss.png')

# plot accuracy curve
plt.cla()
plt.plot(cnn_deep_acc, label='deep')
plt.plot(cnn_middle_acc, label='middle')
plt.plot(cnn_shallow_acc, label='shallow')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('./1_1_2_acc.png')