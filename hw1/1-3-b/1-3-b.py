import tensorflow as tf
import numpy as np   
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import scipy.sparse as sparse
import matplotlib.pyplot as plt

epochs = 100



def main():
    
    b_size = [8,64,256,1024,2048,4096,8192,11264,16384,20480]
    #original part with heavy computation
    
    lul = []
    for b in b_size:
        with open("1-3-b_"+str(b)+".pickle","rb") as handle:
            lul.append(pickle.load(handle))
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    hs = []
    for result in lul:
        loss_train.append(result[0][0][0])
        loss_test.append(result[0][0][1])
        acc_train.append(result[0][0][2])
        acc_test.append(result[0][0][3])
        hs.append(result[0][1])
    #sharpness = [3.9127638,1.0122977,1.0499588,0.7847615,2.480652,3.47683,3.714573,3.8149803,3.8149803,4.359922]
    
    


    
    sharpness = []
    for hessian in hs:
        eigs = []
        for h_layer in hessian:
            print(h_layer.shape)
            val, vec = sparse.linalg.eigs(h_layer,k=1)
            print(val[0])
            eigs.append(val)
        k = eigs.index(max(eigs))
        v = np.linalg.norm(hessian[k],2)
        print(v)
        sharpness.append(v)
    
    

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(b_size, loss_train, 'r--',label="train")
    ax1.plot(b_size, loss_test, 'r-',label="test")

    
    ax2.semilogx(b_size, sharpness, 'b-')

    ax1.set_xlabel('batch size')
    ax1.set_ylabel('loss', color='r')
    ax2.set_ylabel('sharpness', color='b')
    ax1.legend()
    
    plt.savefig("1-3-b_loss.png")
    plt.close()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(b_size, acc_train, 'r--',label="train")
    ax1.plot(b_size, acc_test, 'r-',label="test")

    
    ax2.semilogx(b_size, sharpness, 'b-')

    ax1.set_xlabel('batch size')
    ax1.set_ylabel('acc', color='r')
    ax2.set_ylabel('sharpness', color='b')
    ax1.legend()
    
    plt.savefig("1-3-b_acc.png")
    plt.close()
    
    '''
    with open("1-3-b.pickle","rb") as handle:
        re = pickle.load(handle)
    
    for k in re:
        print(len(k))
    loss_train = re[0]
    loss_test = re[1]
    acc_train = re[2]
    acc_test = re[3]
    sharpness = re[4]
    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(b_size, loss_train, 'r--',label="train")
    ax1.plot(b_size, loss_test, 'r-',label="test")

    
    ax2.semilogx(b_size, sharpness, 'b-')

    ax1.set_xlabel('batch size')
    ax1.set_ylabel('loss', color='r')
    ax2.set_ylabel('sharpness', color='b')
    ax1.legend()
    
    plt.savefig("1-3-b_loss.png")
    plt.close()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(b_size, acc_train, 'r--',label="train")
    ax1.plot(b_size, acc_test, 'r-',label="test")

    
    ax2.semilogx(b_size, sharpness, 'b-')

    ax1.set_xlabel('batch size')
    ax1.set_ylabel('acc', color='r')
    ax2.set_ylabel('sharpness', color='b')
    ax1.legend()
    
    plt.savefig("1-3-b_acc.png")
    plt.close()
    '''
  



    


    

    
                

    
    



if __name__ == "__main__":
    main()    