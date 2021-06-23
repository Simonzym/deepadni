#build RNN for extracted features
import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import nibabel as nib
import random
import pickle
import tensorflow.keras.backend as K
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

def brier(y_true, y_pred):
    return K.mean(K.sum(K.pow(y_true - y_pred, 2), axis = 1))

#build model
def build_rnn(alpha = 0.01, lr = 0.001):
    
    rnn_model = Sequential()
    rnn_model.add(Masking(mask_value=-1, input_shape = (9, 128)))
    rnn_model.add(Bidirectional(GRU(256, return_sequences=True)))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Bidirectional(GRU(128)))
    rnn_model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(alpha)))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Dense(2, activation = 'softmax'))
    opt = Adam(lr = lr)
    rnn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', tf.keras.metrics.AUC(), brier])
    return rnn_model

#turning dictionary of extracted features to (9, 128)
#read dictionary first
def get_seq(folder):
    
    a_file = open(''.join(['Code/Info/SimGraph/', folder, '/train.pkl']), 'rb')
    train_ex = pickle.load(a_file)
    a_file.close()
    
    a_file = open(''.join(['Code/Info/SimGraph/', folder, '/test.pkl']), 'rb')
    test_ex = pickle.load(a_file)
    a_file.close()
    
    train_gid = list(train_ex.keys())
    test_gid = list(test_ex.keys())
    
    train_seq = []
    test_seq = []
    
    for gid in train_gid:
        
        image_gid = train_ex[gid]
        num_img = image_gid.shape[0]
        
        if num_img < 9:
            sup_img = np.zeros((9 - num_img, 128)) - 1
            image_gid = np.vstack([image_gid, sup_img])
        
        train_seq.append(list(image_gid))
        
    for gid in test_gid:
        
        image_gid = test_ex[gid]
        num_img = image_gid.shape[0]
        
        if num_img < 9:
            sup_img = np.zeros((9 - num_img, 128)) - 1
            image_gid = np.vstack([image_gid, sup_img])
        
        test_seq.append(list(image_gid))
    
    return np.array(train_seq), np.array(test_seq)

#read nodes
def convert_seq(folder, sim_turn, set_type = 'train'):
    
    path = ''.join(['Code/Info/SimGraph/', folder, '/sim', str(sim_turn), '/', set_type, '/nodes.csv'])
    graph_path = ''.join(['Code/Info/SimGraph/', folder,'/sim', str(sim_turn), '/', set_type, '/graphs.csv'])
    
    dataset = pd.read_csv(path)
    status = pd.read_csv(graph_path)
    
    all_gid = list(set(dataset['graph_id']))
    seq = []
    y = []
    for gid in all_gid:
        
        image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
        image_gid = np.delete(image_gid, -1, axis=1)
        num_img = image_gid.shape[0]
        if num_img < 9:
            sup_img = np.zeros((9 - num_img, 128)) - 1
            image_gid = np.vstack([image_gid, sup_img])
        
        y_loc = list(status['graph_id']).index(gid)
        y.append(status['label'][y_loc])
        seq.append(list(image_gid))
        
    return np.array(seq), np.array(y)

sims = [7]
for num_sim in sims:
    for sim_turn in range(46, 101):
        
    #get information of extracted features and final outcome
        sim_run = ''.join(['graphSim', str(num_sim)])
        train_seq, train_y = convert_seq(sim_run, sim_turn, 'train')
        test_seq, test_y = convert_seq(sim_run, sim_turn, 'test')
        
        
        #build rnn 
        sim_rnn = build_rnn(alpha = 0, lr = 0.001)
    
        
        #train rnn
        sim_rnn_fit = sim_rnn.fit([train_seq], to_categorical(train_y), 
                                batch_size = 40, epochs = 100, shuffle = True,
                                validation_data = ([test_seq], to_categorical(test_y)), verbose = 0)
        
        hist_df = pd.DataFrame(sim_rnn_fit.history)
        hist_csv_file = ''.join(['Code/Info/SimGraph/graphSim', str(num_sim), '/sim', str(sim_turn), '/rnn_results.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            