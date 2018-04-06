import numpy as np 
import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA


def main():
    
    all = []
    with open("1-2-2.pickle","rb") as handle:
        all.append(pickle.load(handle))




    #print(len(lul))
    #print(type(lul[0]))
    weights = []    
    loss = []
    norm = []
    count = []
    for a in all:
        for i in range(len(a)):
            weights.append(a[i][0])
            loss.append(a[i][2])
            norm.append(a[i][3])
            count.append(a[i][1])
    weights = np.array(weights)
    weights = weights.reshape((-1,weights.shape[1]))
    weights_first = weights[:,:5]
    pca = PCA(n_components = 2, whiten = True, svd_solver = "randomized").fit(weights)
    pca_f = PCA(n_components = 2, whiten = True, svd_solver = "randomized").fit(weights_first)
    weights_2 = pca.transform(weights)
    weights_2_f = pca_f.transform(weights_first)
    '''
    color = ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
    for i in range(8):
        plt.scatter(weights_2[i*6666:(i+1)*6666, 0], weights_2[i*6666:(i+1)*6666,1],cmap = loss[i*6666:(i+1)*6666])
    plt.title("all layers")
    plt.savefig("weights_all.png")
    plt.close()

    for i in range(8):
        plt.scatter(weights_2_f[i*6666:(i+1)*6666, 0], weights_2_f[i*6666:(i+1)*6666,1])
    plt.title("first layer")
    plt.savefig("weights_first.png")
    plt.close()
    
    '''
    k = [0]
    for i in k:
        plt.plot(count[i*6666:(i+1)*6666],norm[i*6666:(i+1)*6666]) 
        plt.savefig("norm"+str((i+1))+".png")
        plt.close()
        plt.plot(count[i*6666:(i+1)*6666],loss[i*6666:(i+1)*6666]) 
        plt.savefig("loss"+str((i+1))+".png")
        plt.close()  

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(count[i*6666:(i+1)*6666], norm[i*6666:(i+1)*6666], 'r-',label="g_norm")
        
        
        ax2.plot(count[i*6666:(i+1)*6666], loss[i*6666:(i+1)*6666], 'b-',label="loss")

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('g_norm', color='r')
        ax2.set_ylabel('loss', color='b')
        plt.legend()
        
        plt.savefig("1-2-2.png")
        plt.close()
          
            
            

if __name__ == '__main__':
    main()
