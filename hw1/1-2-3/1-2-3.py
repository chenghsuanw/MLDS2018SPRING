import numpy as np 
import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pickle

epoch = 10000

def main():
    #original processing code with all the result
    
    with open("100timestest_gradients.pickle","rb") as handle:
        grad = pickle.load(handle)
    with open("100timestest_loss.pickle","rb") as handle:
        loss = pickle.load(handle)
    with open("100timestest_hessians.pickle","rb") as handle:
        hess = pickle.load(handle)
    print(hess)
    print((hess[0][-1]).shape)
    loss_all = []
    hess_all = []
    for i in range(len(hess)):
        
        loss_all.append(loss[i][-1])
        w,_ = np.linalg.eig(hess[i][-1])
        w[w>0] = 1
        w[w <= 0] = 0
        hess_all.append(np.sum(w)/w.shape[0])
    
   
   

    plt.scatter(hess_all, loss_all)
    plt.savefig("1-2-3.png")  

    
            
            

if __name__ == '__main__':
    main()
