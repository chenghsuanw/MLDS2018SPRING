import numpy as np
import tensorflow as tf
import argparse
import pickle
from model_seq2seq import s2s
import gensim
import copy
import sys




def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_label', default='./mlds_hw2_2_data/clr_conversation.txt')
    parser.add_argument('--test_label', default='./mlds_hw2_2_data/test_input.txt')
    parser.add_argument('--pretrained_w2v', default='./glove.txt')

    args = parser.parse_args()
    return args

def load_data(path):
    with open(path) as f:
        content = f.readlines()
    train_data = []
    result = []
    lines = []
    last = None
    for c in content:
        if "+++$+++" in c:
            last = None
            train_data.append(result[1:])
            result = []
        else:
            current = c.rstrip('\n')
            current_list = current.split(" ")
            lines.append(current_list)
            result.append([last, current_list])
            last = copy.deepcopy(current_list)    
    train_data.append(result[1:])
    return [j for i in train_data for j in i], lines

def load_data_test(path):
    with open(path) as f:
        content = f.readlines()
    train_data = []
    result = []
    lines = []
    last = None
    for c in content:
       
        current = c.rstrip('\n')
        current_list = current.split(" ")
        lines.append(current_list)

    
    return lines
        
def build_custom_dict(model):
    w2i = dict()
    i2v = []
    i2w = dict()
    
    c = 0
    for k, _ in model.wv.vocab.items():
        w2i[k] = c
        i2w[c] = k
        c += 1
        v = model.wv[k]
        i2v.append(v)

    i2v = np.array(i2v)
    
    return w2i, i2w, i2v

def transfer_data_to_index_based(w2i,training_data):
    for i in range(len(training_data)):
        for j in range(len(training_data[i])):
            for k in range(len(training_data[i][j])):
                word = training_data[i][j][k]
                index = w2i[word]
                training_data[i][j][k] = index
            #training_data[i][j] = np.array(training_data[i][j])
        #training_data[i] = np.array(training_data[i])
    #training_data = np.array(training_data)
    
    return training_data


    



        


        


def main(argv):
    parameters = dict()
    parameters["word_embedding_dimension"] = 100
    parameters["RNN_hidden_dimension"] = 512
    parameters["input_length"] = 12 
    parameters["maxlen_of_speech"] = 12
    parameters["batch_size"] = 100
    
    

    

    
    
    #args = arg_parse()
    #train_data, lines = load_data(args.train_label)
    #lines.append(["<PAD>"]*3)
    #lines.append(["<UNK>"]*3)
    #lines.append(["<BOS>"]*3)
    #lines.append(["<EOS>"]*3)

    test = load_data_test(argv[0])
    
    '''
    model = gensim.models.Word2Vec(lines,min_count=3,size=100)
    model.save("word2vec.model")
    model = gensim.models.Word2Vec.load("word2vec.model")
    w2i, i2w, i2v = build_custom_dict(model)
    

    
    
    
    with open('w2i.pickle', 'wb') as handle:
        pickle.dump(w2i, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('i2w.pickle', 'wb') as handle:
        pickle.dump(i2w, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    with open('i2v.pickle', 'wb') as handle:
        pickle.dump(i2v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###comment here
    '''

    
    with open("w2i.pickle","rb") as handle:
        w2i = pickle.load(handle)
    
    with open("i2v.pickle","rb") as handle:
        i2v = pickle.load(handle)
    
    with open("i2w.pickle","rb") as handle:
        i2w = pickle.load(handle)
    '''
    print(w2i["<PAD>"])
    print(w2i["<UNK>"])
    print(w2i["<BOS>"])
    print(w2i["<EOS>"])
    '''
    '''
    train_data = transfer_data_to_index_based(w2i, train_data)

    with open('train_data.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    with open("train_data.pickle","rb") as handle:
        train_data = pickle.load(handle)
    '''
    
    zeros = 0
    count = 0
    parameters["voc_size"] = i2v.shape[0]
    model = s2s(parameters,i2v)
    model.train(0,None,w2i,parameters["batch_size"])


    replies = model.predict(100,test,w2i)
    #replies = model.predict_beam_search(parameters["batch_size"],test,w2i)




    s_words = ["<BOS>","<EOS>","<PAD>","<UNK>"]
    result = []
    for reply in replies:
        rep = []
        for word_index in reply:
            word = i2w[word_index]
            if word not in s_words:
                rep.append(word)
        lines = " ".join(rep)
        '''
        if len(lines) == 0:
            lines = "nothing"
            zeros += 1
        '''
        print(" ".join(test[count]))
        print(lines)
        print("------------------------------------------")
        
        result.append(lines)
        count += 1


    text_file = open(argv[1], "w")
    for r in result:
        text_file.write("%s\n" % r)
    text_file.close()
    
    
    
    





    
             
                    




if __name__ == '__main__':
    main(sys.argv[1:])


