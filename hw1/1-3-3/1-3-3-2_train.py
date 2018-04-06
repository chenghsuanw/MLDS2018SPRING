import tensorflow as tf
import numpy as np   
from tensorflow.examples.tutorials.mnist import input_data
import pickle


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

    W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,fil1_n],stddev=0.1),name="WC1")
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [fil1_n]),name="BC1")

    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides = [1,1,1,1], padding = 'SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1,ksize = [1,2,2,1],strides = [1,2,2,1], padding = "SAME")



    W_conv2 = tf.Variable(tf.truncated_normal([5,5,fil1_n,fil2_n],stddev=0.1),name="WC2")
    b_conv2 = tf.Variable(tf.constant(0.1, shape = [fil2_n]),name="BC2")

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides = [1,1,1,1], padding = 'SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize = [1,2,2,1],strides = [1,2,2,1], padding = "SAME")

    W_fc1 = tf.Variable(tf.truncated_normal([7*7*fil2_n, fc1_out_n], stddev=0.1),name="WF1")
    b_fc1 = tf.Variable(tf.constant(0.1, shape = [fc1_out_n]),name="BF1")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*fil2_n])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal([fc1_out_n, 10], stddev=0.1),name="WF2")
    b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]),name="BF2")
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0)),
                                              reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    
    
    re = []
    para_count = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    #builder = tf.saved_model.builder.SavedModelBuilder("./")
    with tf.Session(config = config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        
        for i in range(20):
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
        print(loss)
        saver.save(sess, "./model/model"+str(batch_size))
        
    return para_count, re


def main():
    
    result = []
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    b_size = [8,16,64,256,512,1024,2048,4096,8192,16384]

    for b in b_size:
        _, _ = train(mnist, 16,32,128,b)
        tf.reset_default_graph()
        
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
    main()    