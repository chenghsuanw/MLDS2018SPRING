import tensorflow as tf
import numpy as np   
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import matplotlib.pyplot as plt
import math

epochs = 100

def main():
    
    result = []

    train = np.ones((10000,2))
    for i in range(10000):
        x = (i + 1)/ 10000
        ##x = (2 * i + 1)/10000
        train[i,0] = x
        train[i,1] = math.sin(x * math.pi * 5) / ( 5 * math.pi * x)	
        ##train[i,1] = x * x
    
    
    
    
    with open("last10.pickle","rb") as handle:
        weights = pickle.load(handle)
    final_weights = weights[0]
    perturb = np.arange(-0.001, 0.0011,0.0001)
    to_tune = [0,5,10,60,70,170,180,230,235,240,241]
    result = []
    for p in perturb:
        line = []
        
        for t in range(10):
            
            x = tf.placeholder(tf.float32, [None, 1], name = "Traindata")
            y = tf.placeholder(tf.float32, [None, 1], name = "label")
        
            graph = tf.get_default_graph()
            new_w = final_weights
            print(p)
            
            for i in range(to_tune[t],to_tune[t+1]):
                new_w[i] += p
            
            parameters = tf.Variable(initial_value=new_w,name="super")
            

            with tf.name_scope("W1"):
                idx_from = 0 
                weights1 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[5, 1]), [1, 5])
                idx_from = idx_from + 5
                biases1 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[5, 1]), [5]) # tf.Variable(tf.truncated_normal([n_hidden]))
                hidden1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)

            with tf.name_scope("W2"):
                idx_from = 10
                weights2 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[50, 1]), [5, 10])
                idx_from = idx_from + 50
                biases2 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[10, 1]), [10]) # tf.Variable(tf.truncated_normal([n_hidden]))
                hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

            with tf.name_scope("W3"):
                idx_from = 70 
                weights3 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[100, 1]), [10, 10])
                idx_from = idx_from + 100
                biases3 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[10, 1]), [10]) # tf.Variable(tf.truncated_normal([n_hidden]))
                hidden3 = tf.nn.relu(tf.matmul(hidden2, weights3) + biases3)

            with tf.name_scope("W4"):
                idx_from = 180
                weights4 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[50, 1]), [10, 5])
                idx_from = idx_from + 50
                biases4 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[5, 1]), [5]) # tf.Variable(tf.truncated_normal([n_hidden]))
                hidden4 = tf.nn.relu(tf.matmul(hidden3, weights4) + biases4)

            with tf.name_scope("W5"):
                idx_from = 235
                weights5 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[5, 1]), [5, 1])
                idx_from = idx_from + 5
                biases5 = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[1, 1]), [1]) 
                pred = tf.matmul(hidden4, weights5) + biases5
            
            with tf.name_scope("Loss"):
                loss = tf.reduce_mean(tf.square(pred - y))


            
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.allow_growth = True
            
            with tf.Session(config = config) as sess:
                
                sess.run(tf.global_variables_initializer())
                
                c = sess.run(loss,feed_dict={x: train[:,0].reshape((10000,1)), y: train[:,1].reshape((10000,1))})
                print(c)
            line.append(c)
            graph = tf.reset_default_graph()
        result.append(line)
    
    
    
    perturb = np.arange(-0.001, 0.0011,0.0001)
    
    
    for i in range(len(result)):
        result[i] = np.array(result[i])
    result = np.array(result)
    for i in range(10):
        plt.plot(perturb,result[:,i])    
    
    
    plt.savefig("1-2-b-2.png")
    plt.close()

    
  



    


    

   

    
    



if __name__ == "__main__":
    main()    