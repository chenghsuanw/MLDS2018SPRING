import tensorflow as tf
import numpy as np   
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import matplotlib.pyplot as plt

epochs = 100

def main():
    #part for loading and computing
    
    result = []
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    
    
    weights_a = []
    
    with tf.Session() as sess:  
        saver = tf.train.import_meta_graph("./model/model64.meta")
        saver.restore(sess,"./model/model64")
        graph = tf.get_default_graph()
        
        
        weights_a.append(graph.get_tensor_by_name("WC1:0").eval())
        weights_a.append(graph.get_tensor_by_name("BC1:0").eval())
        weights_a.append(graph.get_tensor_by_name("WC2:0").eval())
        weights_a.append(graph.get_tensor_by_name("BC2:0").eval())
        weights_a.append(graph.get_tensor_by_name("WF1:0").eval())
        weights_a.append(graph.get_tensor_by_name("BF1:0").eval())
        weights_a.append(graph.get_tensor_by_name("WF2:0").eval())
        weights_a.append(graph.get_tensor_by_name("BF2:0").eval())

        print(graph.get_tensor_by_name("BC1:0").eval())
    
    
    
    graph = tf.reset_default_graph()
    
    weights_b = []
    

    with tf.Session() as sess:  
        saver = tf.train.import_meta_graph("./model/model1024.meta")
        saver.restore(sess,"./model/model1024")
        graph = tf.get_default_graph()

        weights_b.append(graph.get_tensor_by_name("WC1:0").eval())
        weights_b.append(graph.get_tensor_by_name("BC1:0").eval())
        weights_b.append(graph.get_tensor_by_name("WC2:0").eval())
        weights_b.append(graph.get_tensor_by_name("BC2:0").eval())
        weights_b.append(graph.get_tensor_by_name("WF1:0").eval())
        weights_b.append(graph.get_tensor_by_name("BF1:0").eval())
        weights_b.append(graph.get_tensor_by_name("WF2:0").eval())
        weights_b.append(graph.get_tensor_by_name("BF2:0").eval())

        print(graph.get_tensor_by_name("BC1:0").eval())

    
    x_train, y_train = mnist.train.next_batch(10000)
    x_test, y_test = mnist.test.next_batch(10000)

    alphas = [i/100 for i in range(-100,200)]
    for alpha in alphas:
        weights_new = []
        for i in range(len(weights_a)):
            new_weight = (1-alpha) * weights_a[i] + alpha * weights_b[i]
            weights_new.append(new_weight)
        graph = tf.reset_default_graph()
        ys = tf.placeholder(tf.float32, [None, 10])
        xs = tf.placeholder(tf.float32, [None, 784])
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(xs, [-1, 28, 28, 1])

        W_conv1 = tf.get_variable(initializer=weights_new[0],name = "lul")

        b_conv1 = tf.get_variable(initializer=weights_new[1],name = "lul1")

        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides = [1,1,1,1], padding = 'SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1,ksize = [1,2,2,1],strides = [1,2,2,1], padding = "SAME")



        W_conv2 = tf.get_variable(initializer=weights_new[2],name = "lul2")
        b_conv2 = tf.get_variable(initializer=weights_new[3],name = "lul3")

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides = [1,1,1,1], padding = 'SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2,ksize = [1,2,2,1],strides = [1,2,2,1], padding = "SAME")

        W_fc1 = tf.get_variable(initializer=weights_new[4],name = "lul4")
        b_fc1 = tf.get_variable(initializer=weights_new[5],name = "lul5")
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = tf.get_variable(initializer=weights_new[6],name = "lul6")
        b_fc2 = tf.get_variable(initializer=weights_new[7],name = "lul7")
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0)),
                                              reduction_indices=[1]))
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            
            
            y_pre = sess.run(prediction, feed_dict={xs: x_test, keep_prob: 1})
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_test,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result_test = sess.run(accuracy, feed_dict={xs: x_test, ys: y_test, keep_prob: 1})
            loss_test = sess.run(cross_entropy,feed_dict={xs:x_test , ys: y_test, keep_prob:1})
            y_pre = sess.run(prediction, feed_dict={xs: x_train, keep_prob: 1})
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_train,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result_train = sess.run(accuracy, feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
            loss_train = sess.run(cross_entropy,feed_dict={xs:x_train , ys: y_train, keep_prob:1})
            
            result.append([alpha,loss_train, loss_test, result_train, result_test])

    
   
    alpha = []
    loss_train = []
    loss_test = []
    result_train = []
    result_test = []        
    for k in result:
        alpha.append(k[0])
        loss_train.append(k[1])
        loss_test.append(k[2])
        result_train.append(k[3])
        result_test.append(k[4])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(alpha, loss_train, 'r--',label="train")
    ax1.plot(alpha, loss_test, 'r-',label="test")

    ax2.plot(alpha, result_train, 'b--')
    ax2.plot(alpha, result_test, 'b-')

    ax1.set_xlabel('alpha')
    ax1.set_ylabel('loss', color='r')
    ax2.set_ylabel('acc', color='b')
    plt.legend()
    
    plt.savefig("1-3-3-1.png")
    plt.close()

    
  



    


    

    
    
    



if __name__ == "__main__":
    main()    