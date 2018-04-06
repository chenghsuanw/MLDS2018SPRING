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

    

    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    
    x = tf.placeholder(tf.float32, [None, 1], name = "Traindata")
    y = tf.placeholder(tf.float32, [None, 1], name = "label")

    W1 = tf.Variable(initializer([1,5]), name='Weights1')
    b1 = tf.Variable(initializer([5]), name='Bias1')

    W2 = tf.Variable(initializer([5,10]), name='Weights2')
    b2 = tf.Variable(initializer([10]), name='Bias2')

    W3 = tf.Variable(initializer([10,10]), name='Weights3')
    b3 = tf.Variable(initializer([10]), name='Bias3')
    
    W4 = tf.Variable(initializer([10,10]), name='Weights4')
    b4 = tf.Variable(initializer([10]), name='Bias4')
    
    W5 = tf.Variable(initializer([10,10]), name='Weights5')
    b5 = tf.Variable(initializer([10]), name='Bias5')

    W6 = tf.Variable(initializer([10,10]), name='Weights6')
    b6 = tf.Variable(initializer([10]), name='Bias6')

    W7 = tf.Variable(initializer([10,5]), name='Weights7')
    b7 = tf.Variable(initializer([5]), name='Bias7')
    
    W8 = tf.Variable(initializer([5,1]), name='Weights8')
    b8 = tf.Variable(initializer([1]), name='Bias8')


    with tf.name_scope('Model'):
        layer1 = tf.nn.leaky_relu(tf.matmul(x, W1) + b1)
        layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)
        layer3 = tf.nn.leaky_relu(tf.matmul(layer2, W3) + b3)
        layer4 = tf.nn.leaky_relu(tf.matmul(layer3, W4) + b4)
        layer5 = tf.nn.leaky_relu(tf.matmul(layer4, W5) + b5)
        layer6 = tf.nn.leaky_relu(tf.matmul(layer5, W6) + b6)
        layer7 = tf.nn.leaky_relu(tf.matmul(layer6, W7) + b7)
        pred = tf.matmul(layer7, W8) + b8

    with tf.name_scope("Loss"):
        loss = tf.reduce_mean(tf.square(pred - y))

    with tf.name_scope("Adam"):
        optimize = tf.train.AdamOptimizer()
    
        optimizer = optimize.minimize(loss)
    
        get_grad = optimize.compute_gradients(loss,[W1,W2,W3,W4,W5,W6,W7,W8])

    init = tf.global_variables_initializer()

    tf.summary.scalar("Loss", loss)
    merged_summary_op = tf.summary.merge_all()
    ##print("huehue")
    result = []
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        sess.run(init)

        summary_writer = tf.summary.FileWriter("./record", graph = tf.get_default_graph())

        for e in range(20000):
            avg_loss = 0
            for k in range(100):            
                _, c, summary = sess.run([optimizer, loss, merged_summary_op],feed_dict={x: train[k*100:k*100+100,0].reshape((100,1)), y: train[k*100:k*100+100,1].reshape((100,1))})
                summary_writer.add_summary(summary, e)
                avg_loss += c
            if (e+1) % 50 == 0:
                print("Epoch:", '%04d' % (e+1), "cost=", "{:.9f}".format(avg_loss / 100))
            
            if (e+1) % 3 == 0:
                weight = []
                for i in range(1,9):
                    weight_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Weights"+str(i))[0]
                    weight_var_value = sess.run(weight_var)
                    weight.append(weight_var_value.reshape([-1,1]))
                weights = np.concatenate(weight)
                

                grads = sess.run(get_grad,feed_dict = {x:train[:,0].reshape((10000,1)), y:train[:,1].reshape((10000,1))})
                grads_all = 0
                for i in range(8):
                    grads_all += (grads[i][0] ** 2).sum()
                grad_norm = grads_all ** 0.5
                re = [weights, e, avg_loss, grad_norm]
                result.append(re)
                
                
            
        with open('1-2-2.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        feed_dict = {x: train[:,0].reshape(10000,1)}
        lul = sess.run(pred, feed_dict)
        plt.scatter(train[:,0],lul)
        plt.savefig("resul.png")

     
            

if __name__ == '__main__':
    main()
