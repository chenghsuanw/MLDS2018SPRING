import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core
import re
from tqdm import tqdm
import pickle
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def process_data(label, w2i,max_length = 12):
    
    s_list = label
    
    result_in = []
    result_out = []
    result_in.append(w2i["<BOS>"])
    for i in range(max_length-1):
        
        if i < len(s_list):
            if s_list[i] in w2i:
                result_in.append(w2i[s_list[i]])
            else:
                result_in.append(w2i["<UNK>"])
            
        else:
            result_in.append(w2i["<PAD>"])

    for i in range(max_length-1):
        
        if i < len(s_list):
            if s_list[i] in w2i:
                result_out.append(w2i[s_list[i]])
            else:
                result_out.append(w2i["<UNK>"])
            
        else:
            result_out.append(w2i["<PAD>"])
            
    result_out.append(w2i["<EOS>"])
    result_in = np.array(result_in, dtype=np.int32)
    result_out = np.array(result_out, dtype=np.int32)
    return result_in , result_out

def process_data_train(label, w2i,max_length = 12):
    
    s_list = label
    
    result_in = []
    for i in range(max_length):
        
        if i < len(s_list):
            if s_list[i] in w2i:
                result_in.append(w2i[s_list[i]])
            else:
                result_in.append(w2i["<UNK>"])
            
        else:
            result_in.append(w2i["<PAD>"])

    
    result_in = np.array(result_in, dtype=np.int32)
    
    return result_in

def build_batch(batch_size,train_data,w2i):
    
    num_data = len(train_data)
    index = np.random.randint(0, num_data - 1,size =  batch_size)
    
    batch_data = []
    for i in index:
        gg = train_data[i][0]
        gg = process_data_train(gg,w2i)
        batch_data.append(gg)
       
    batch_data = np.array(batch_data)
    batch_data = batch_data.T
    

    decoder_in = []
    decoder_out = []

    for i in index:
        gg = train_data[i][1]
        d_in,d_out = process_data(gg,w2i)
        decoder_in.append(d_in)
        decoder_out.append(d_out)
    
    decoder_in = np.array(decoder_in)
    decoder_out = np.array(decoder_out)
    decoder_in = decoder_in.T

    return batch_data, decoder_in, decoder_out

def build_batch_100(data,w2i):
    
    batch_data = []
    for i in range(len(data)):
        gg = data[i]
        gg = process_data_train(gg,w2i)
        batch_data.append(gg)
    num = len(batch_data)
    if num < 100:
        for i in range(100-num):
            batch_data.append(batch_data[0])
       
    batch_data = np.array(batch_data)
    batch_data = batch_data.T
    

    
    return batch_data
    
    


class s2s():
    
    def __init__(self,parameters,w2v):
        self.frame_dim = parameters["word_embedding_dimension"]
        self.word_dim = parameters["word_embedding_dimension"]
        self.hidden_dim = parameters["RNN_hidden_dimension"]
        self.voc_size = parameters["voc_size"]
        self.frame_num = parameters["input_length"]
        self.speech_len = parameters["maxlen_of_speech"]
        
        
        '''
        print(encoder_in.shape)
        print(decoder_in.shape)
        print(decoder_out.shape)
        exit()
        '''
        
        self.e_in = tf.placeholder(tf.int32, [self.frame_num, 100], name='encoder_in')
        self.d_in_idx = tf.placeholder(tf.int32, [self.speech_len,100], name='decoder_in_idx')
        self.d_out_idx = tf.placeholder(tf.int32, [100, self.speech_len], name='decoder_out_idx')
        self.decoder_lengths = tf.placeholder(tf.int32, [100], name="decoer_length")
        self.dropout_rate = tf.placeholder(tf.float32, shape=[])
        self.W2V = tf.Variable(w2v, name='W')

        self.e_in_e = tf.nn.embedding_lookup(self.W2V, self.e_in)
        self.d_in = tf.nn.embedding_lookup(self.W2V, self.d_in_idx)
        self.sampling_prob = tf.placeholder(tf.float32,shape=[])
        #self.d_in = tf.transpose(self.d_in,perm = [1,0,2])
        

        

        with tf.name_scope("encoder"):
            self.encoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            self.encoder_cell2 = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            self.multi_encoder = tf.nn.rnn_cell.MultiRNNCell([self.encoder_cell, self.encoder_cell2])
            self.multi_encoder = tf.nn.rnn_cell.DropoutWrapper(
                self.multi_encoder, output_keep_prob=1.0 - self.dropout_rate)

            encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                self.multi_encoder, self.e_in_e, time_major=True, dtype=tf.float32)


            

        with tf.name_scope("decoder"):

            

            

            self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            self.decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
                self.decoder_cell, output_keep_prob=1.0 - self.dropout_rate)
            '''
            helper = tf.contrib.seq2seq.TrainingHelper(
                self.d_in, self.decoder_lengths, time_major=True)
            '''
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                self.d_in,
                self.decoder_lengths,
                self.W2V,
                sampling_probability=self.sampling_prob,
                time_major = True)
            
            self.projection_layer = layers_core.Dense(
                self.voc_size, use_bias=False)

            #where to add attention
            
            self.attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.hidden_dim, self.attention_states,
                memory_sequence_length=None)

            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell, attention_mechanism,
                attention_layer_size=self.hidden_dim)
            
            

            self.init_state = self.decoder_cell.zero_state(100, tf.float32).clone(cell_state=self.encoder_state[-1])
            
            #self.init_state = encoder_state[-1]
            

            decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.init_state,
                output_layer=self.projection_layer)

            outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output

        with tf.name_scope("Loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.d_out_idx, logits=logits)
            #train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)

        

    def train(self,iterations,train_data,w2i,batch_size): 
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, 5)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        
        
        learning_rate_dacay = 1
        sampling_prob_decay = 0.95
        
        
            
        for iters in range(iterations):
            
            batch_data, decoder_in, decoder_out = build_batch(batch_size,train_data,w2i)

            feed_dict = {
            self.e_in: batch_data,
            self.d_out_idx: decoder_out,
            self.d_in_idx: decoder_in,
            self.decoder_lengths: np.ones((batch_size), dtype=int) * self.speech_len,
            learning_rate : 0.001 * learning_rate_dacay,
            self.dropout_rate : 0.5,
            self.sampling_prob : sampling_prob_decay
            }
            
            _, loss_value = sess.run([train_op, self.loss], feed_dict=feed_dict)
            loss = np.mean(loss_value)
        
            
            if iters%100 == 0:
                print("train_loss:", loss, "iterations: ", iters+1)
            if (iters+1)%10000 == 0:
                
                saver.save(sess, "./model/model_epoch_"+str(iters+1))
            if (iters+1)%3000 == 0:
                learning_rate_dacay *= 0.95
            if (iters+1) % 100 == 0:
                sampling_prob_decay *= 0.95
                

            
        if iterations != 0:
            saver.save(sess, "./model/model_epoch_final")

    def predict(self,batch_size, test_data,w2i):
        
        if len(test_data) == 0:
            print("NULL INPUT!!")
            return [[],]
        

       
        # Inference
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.W2V,
            tf.fill([batch_size], 61766), 55029)

        # Inference Decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell, inference_helper, self.init_state,
            output_layer=self.projection_layer)


        # We should specify maximum_iterations, it can't stop otherwise.
        source_sequence_length = self.speech_len
        maximum_iterations = tf.round(tf.reduce_max(source_sequence_length))

        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder, maximum_iterations=maximum_iterations)
        translations = outputs.sample_id

        #tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph("./model/model_epoch_final.meta")
        saver.restore(sess,"./model/model_epoch_final")
        
        recur_times = len(test_data) // 100 + 1 if len(test_data) % 100 != 0 else len(test_data) // 100
        result = []
        for i in range(recur_times):
            start = i*100
            end = min((i+1) * 100, len(test_data))
            batch_data = build_batch_100(test_data[start:end],w2i)

            
            feed_dict = {
                    self.e_in: batch_data,
                    self.dropout_rate : 0
                }

            replies = sess.run([translations], feed_dict=feed_dict)
            for r in replies[0]:
                result.append(r)
            

        
        
        return result[:len(test_data)]

    
        



        

        


        

        

        

        
        

    