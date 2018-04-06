import tensorflow as tf
import numpy as np 
import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pickle

epoch = 10000


    

def main():
    
    train = np.ones((10000,2))
    for i in range(10000):
        x = (i + 1)/ 10000
        ##x = (2 * i + 1)/10000
        train[i,0] = x
        train[i,1] = math.sin(x * math.pi * 5) / ( 5 * math.pi * x)	
        ##train[i,1] = x * x

    

    initializer = tf.contrib.layers.xavier_initializer()
    
    x = tf.placeholder(tf.float32, [None, 1], name = "Traindata")
    y = tf.placeholder(tf.float32, [None, 1], name = "label")
    '''
    W1 = tf.Variable(initializer([1,5]), name='Weights1')
    b1 = tf.Variable(initializer([5]), name='Bias1')

    W2 = tf.Variable(initializer([5,10]), name='Weights2')
    b2 = tf.Variable(initializer([10]), name='Bias2')
    
    W3 = tf.Variable(initializer([10,10]), name='Weights3')
    b3 = tf.Variable(initializer([10]), name='Bias3')
    
    W4 = tf.Variable(initializer([10,5]), name='Weights4')
    b4 = tf.Variable(initializer([5]), name='Bias4')
    
    W5 = tf.Variable(initializer([5,1]), name='Weights5')
    b5 = tf.Variable(initializer([1]), name='Bias5')
    '''
    parameters = tf.Variable(tf.concat([tf.truncated_normal([5, 1]), tf.zeros([5, 1]),
        tf.truncated_normal([50,1]), tf.zeros([10, 1]),
        tf.truncated_normal([100,1]), tf.zeros([10, 1]),
        tf.truncated_normal([50,1]), tf.zeros([5, 1]),
        tf.truncated_normal([5,1]), tf.zeros([1, 1])],0))
    
    print(parameters)
    
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
        
    tvars = tf.trainable_variables()
    dloss_dw = tf.gradients(loss, tvars)[0]
    dim, _ = dloss_dw.get_shape()
    hess = []
    for i in range(dim):
        # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
        dfx_i = tf.slice(dloss_dw, begin=[i,0] , size=[1,1])
        ddfx_i = tf.gradients(dfx_i, parameters)[0] # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
        hess.append(ddfx_i)
    hess = tf.squeeze(hess) 

    init = tf.global_variables_initializer()

    

    lul = []
    with tf.name_scope("Loss_2"):
        for k in get_grad:
            lul.append(tf.reshape(k[0],[-1,1]))
    
        result = tf.concat(lul,0)

        loss2 = tf.reduce_sum(tf.square(result))
    optimizer2 = optimize.minimize(loss2)
    tf.summary.scalar("Loss", loss)
    merged_summary_op = tf.summary.merge_all()
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    final_result_loss = []
    final_result_grad = []
    final_result_hessians = []
    for _ in range(100):
        hessians = []
        losses = []
        gradients = []
        with tf.Session(config = config) as sess:
            sess.run(init)
            for e in range(1000):
                avg_loss = 0
                for k in range(10):            
                    _, c, summary = sess.run([optimizer, loss, merged_summary_op],feed_dict={x: train[k*1000:k*1000+1000,0].reshape((1000,1)), y: train[k*1000:k*1000+1000,1].reshape((1000,1))})
                    
                    avg_loss += c
                
                if (e+1) % 50 == 0:
                    hessians.append(sess.run(hess,feed_dict={x:train[:,0].reshape((10000,1)), y:train[:,1].reshape((10000,1))}))
                    losses.append(sess.run(loss,feed_dict={x:train[:,0].reshape((10000,1)), y:train[:,1].reshape((10000,1))}))
                    
                
            for e in range(1000):
                avg_loss = 0
                for k in range(10):            
                    _, c, summary = sess.run([optimizer2, loss2, merged_summary_op],feed_dict={x: train[k*1000:k*1000+1000,0].reshape((1000,1)), y: train[k*1000:k*1000+1000,1].reshape((1000,1))})
                    
                    avg_loss += c
                
                
                if (e+1) % 50 == 0:
                    hessians.append(sess.run(hess,feed_dict={x:train[:,0].reshape((10000,1)), y:train[:,1].reshape((10000,1))}))
                    gradients.append(sess.run(loss2,feed_dict={x:train[:,0].reshape((10000,1)), y:train[:,1].reshape((10000,1))}))
                    losses.append(sess.run(loss,feed_dict={x:train[:,0].reshape((10000,1)), y:train[:,1].reshape((10000,1))}))
                    
                    
                    
        final_result_hessians.append(hessians)
        final_result_loss.append(losses)
        final_result_grad.append(gradients)
                
            
    with open('100timestest_hessians.pickle', 'wb') as handle:
        pickle.dump(final_result_hessians, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('100timestest_loss.pickle', 'wb') as handle:
        pickle.dump(final_result_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('100timestest_gradients.pickle', 'wb') as handle:
        pickle.dump(final_result_grad, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #feed_dict = {x: train[:,0].reshape(10000,1)}
        #lul = sess.run(pred, feed_dict)
        #plt.scatter(train[:,0],lul)
        #plt.savefig("resul.png")


     
            

if __name__ == '__main__':
    main()
