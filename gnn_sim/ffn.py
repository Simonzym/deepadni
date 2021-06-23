#build RNN for extracted features
#run get_data() first
import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import nibabel as nib
import random
import pickle
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

def brier(y_true, y_pred):
    return K.mean(K.sum(K.pow(y_true - y_pred, 2), axis = 1))

def build_ffn(alpha = 0.01, lr = 0.001):
    
    ffn_model = Sequential()
    ffn_model.add(Input(shape = (128)))
    ffn_model.add(Dense(256, activation = 'relu', kernel_regularizer = l2(alpha)))
    ffn_model.add(Dropout(0.2))
    ffn_model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(alpha)))
    ffn_model.add(Dropout(0.2))
    ffn_model.add(Dense(2, activation = 'softmax'))
    opt = Adam(lr = lr)
    ffn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', tf.keras.metrics.AUC(), brier])
    
    return ffn_model

def get_last(folder, sim_turn, set_type = 'train'):
    
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
        y_loc = list(status['graph_id']).index(gid)
        y.append(status['label'][y_loc])
        seq.append(list(image_gid[num_img-1]))
        
    return np.array(seq), np.array(y)

sims = [2, 3, 4, 7]
for num_sim in sims:
    for sim_turn in range(51, 101):
    
        sim_run = ''.join(['graphSim', str(num_sim)])
        train_seq, train_y = get_last(sim_run, sim_turn, 'train')
        test_seq, test_y = get_last(sim_run, sim_turn, 'test')
        
        
        sim_ffn = build_ffn(alpha = 0, lr = 0.001)
        sim_fit = sim_ffn.fit([train_seq], to_categorical(train_y),
                                  batch_size = 32, epochs = 100,  shuffle = True,
                                  validation_data = ([test_seq], to_categorical(test_y)), verbose = 3)
        
        hist_df = pd.DataFrame(sim_fit.history)
        hist_csv_file = ''.join(['Code/Info/SimGraph/graphSim', str(num_sim), '/sim', str(sim_turn), '/ffn_results.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
    