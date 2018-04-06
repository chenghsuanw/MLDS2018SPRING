import tensorflow as tf
import numpy as np 
import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pickle

epoch = 10000


    

def main():
    #training part
    
    train = np.ones((10000,2))
    for i in range(10000):
        x = (i + 1)/ 10000
        ##x = (2 * i + 1)/10000
        train[i,0] = x
        train[i,1] = math.sin(x * math.pi * 5) / ( 5 * math.pi * x)	
        ##train[i,1] = x * x
    
    x = tf.placeholder(tf.float32, [None, 1], name = "Traindata")
    y = tf.placeholder(tf.float32, [None, 1], name = "label")
    with open("b3_weights.pickle","rb") as handle:
        weights = pickle.load(handle)

    start = np.array(weights[0])
    end = np.array(weights[1])

    diff = end - start
    lul = []

    for i in range(0,10001):
        temp_weight = start + (diff / 10000) * i

        x = tf.placeholder(tf.float32, [None, 1], name = "Traindata")
        y = tf.placeholder(tf.float32, [None, 1], name = "label")

        graph = tf.get_default_graph()

        parameters = tf.Variable(initial_value=temp_weight,name="super")
    
    
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

        with tf.name_scope("Adam"):
            optimize = tf.train.AdamOptimizer()
            optimizer = optimize.minimize(loss)
            get_grad = optimize.compute_gradients(loss,parameters)
            
        

        init = tf.global_variables_initializer()

        

        
        
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        
        
        
        with tf.Session(config = config) as sess:
            
            sess.run(init)
            c = sess.run(loss,feed_dict={x: train[:,0].reshape((10000,1)), y: train[:,1].reshape((10000,1))})
            print(c)
            lul.append(c)
        graph = tf.reset_default_graph()


  
    x = [i for i in range(10001)]
    plt.plot(x, lul,label = "loss")
    plt.legend()
    plt.savefig("1-2-b-3.png")
            
                    
                  
                


                    
                    
                    
        
                
    
     
            

if __name__ == '__main__':
    main()
