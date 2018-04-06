import tensorflow as tf
import numpy as np   
from tensorflow.examples.tutorials.mnist import input_data
import pickle

import sys

epochs = 100

def train(mnist, fil1_n, fil2_n,fc1_out_n, batch_size):
    
    

    xs = tf.placeholder(tf.float32, [None, 784])
    '''
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(xs, W) + b
    '''
    #x_train, y_train = mnist.train.next_batch(10000)
    #x_test, y_test = mnist.test.next_batch(10000)

    ys = tf.placeholder(tf.float32, [None, 10])

    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    W_conv1 = tf.Variable(tf.truncated_normal([5*5*1*fil1_n],stddev=0.1),name="WC1")
    #W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,fil1_n],stddev=0.1),name="WC1")
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [fil1_n]),name="BC1")
    W_conv1_r = tf.reshape(W_conv1,[5,5,1,fil1_n])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1_r,strides = [1,1,1,1], padding = 'VALID') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1,ksize = [1,2,2,1],strides = [1,2,2,1], padding = "SAME")

    W_conv2 = tf.Variable(tf.truncated_normal([5*5*fil1_n*fil2_n],stddev=0.1),name="WC2")
    #W_conv2 = tf.Variable(tf.truncated_normal([5,5,fil1_n,fil2_n],stddev=0.1),name="WC2")
    b_conv2 = tf.Variable(tf.constant(0.1, shape = [fil2_n]),name="BC2")
    W_conv2_r = tf.reshape(W_conv2,[5,5,fil1_n,fil2_n])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2_r,strides = [1,1,1,1], padding = 'VALID') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize = [1,2,2,1],strides = [1,2,2,1], padding = "SAME")


    W_fc1 = tf.Variable(tf.truncated_normal([4*4*fil2_n*fc1_out_n], stddev=0.1),name="WF1")
    #W_fc1 = tf.Variable(tf.truncated_normal([7*7*fil2_n, fc1_out_n], stddev=0.1),name="WF1")
    b_fc1 = tf.Variable(tf.constant(0.1, shape = [fc1_out_n]),name="BF1")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*fil2_n])
    W_fc1_r = tf.reshape(W_fc1,[4*4*fil2_n, fc1_out_n])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1_r) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal([fc1_out_n*10], stddev=0.1),name="WF2")
    #W_fc2 = tf.Variable(tf.truncated_normal([fc1_out_n, 10], stddev=0.1),name="WF2")
    b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]),name="BF2")
    W_fc2_r = tf.reshape(W_fc2,[fc1_out_n, 10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_r) + b_fc2)



    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0)),
                                              reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    
        
    h = tf.hessians(cross_entropy, [W_conv1, b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2])
    
    
    
    re = []
    para_count = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print(para_count)
    hessians = None
    #builder = tf.saved_model.builder.SavedModelBuilder("./")
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        
        for i in range(150):
            for _ in range(55000//batch_size):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, loss = sess.run([train_step,cross_entropy], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:0.5})
            
            '''
            if (i+1) % 1 == 0:
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
                print(loss_train, loss_test, result_train, result_test)
                re.append([loss_train, loss_test, result_train, result_test])
            '''
        batch_xs, batch_ys = mnist.train.next_batch(500)
        batch_xst, batch_yst = mnist.test.next_batch(500)
        hessians = sess.run(h,feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:1})
        
        y_pre = sess.run(prediction, feed_dict={xs: batch_xst, keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(batch_yst,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result_test = sess.run(accuracy, feed_dict={xs: batch_xst, ys: batch_yst, keep_prob: 1})
        loss_test = sess.run(cross_entropy,feed_dict={xs:batch_xst , ys: batch_yst, keep_prob:1})

        y_pre = sess.run(prediction, feed_dict={xs: batch_xs, keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(batch_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result_train = sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
        loss_train = sess.run(cross_entropy,feed_dict={xs:batch_xs , ys: batch_ys, keep_prob:1})
        print(loss_train, loss_test, result_train, result_test)
        things = [loss_train, loss_test, result_train, result_test]
        
        
        
        saver.save(sess, "./model/model"+str(batch_size))
        
    return things, hessians


def main(argv):
    
    result = []
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    b_size = [int(argv[0])]
    #b_size = [16384]

    for b in b_size:
        t, hh = train(mnist, 22,22,32,b)
        tf.reset_default_graph()
        result.append([t,hh])
    
    with open('1-3-b_'+str(argv[0])+'.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    '''
    c = 1
    for f1s in f1:
        for f2s in f2:
            for fcs in fc:
                print("Training No."+str(c)+" loop!")
                p, re = train(mnist, f1s,f2s,fcs)
                print([p,re[-1]])
                result.append([p,re])
                tf.reset_default_graph()
                c += 1
    '''
    
                

    
    



if __name__ == "__main__":
    main(sys.argv[1:])    