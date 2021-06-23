#build RNN for extracted features
import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import nibabel as nib
import random
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

#build model
def adas_rnn(lr = 0.001, alpha = 0.02, size = 258):
    
    rnn_model = Sequential()
    rnn_model.add(Masking(mask_value=-1, input_shape = (9, size)))
    rnn_model.add(Bidirectional(GRU(256, return_sequences=True, kernel_regularizer = l2(alpha))))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Bidirectional(GRU(128, kernel_regularizer = l2(alpha))))
    rnn_model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(alpha)))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Dense(1, activation = 'linear', kernel_regularizer = l2(alpha)))
    opt = Adam(lr = lr)
    rnn_model.compile(loss = 'mse', optimizer = opt, metrics = ['mse'])
    return rnn_model

def convert_seq_adas(num_cv, set_type = 'train'):
    
    path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/', set_type, '_nodes.csv'])
    graph_path = 'Code/Info/graphDIF12/graphs.csv'
    
    dataset = pd.read_csv(path)
    status = pd.read_csv(graph_path)
    
    all_gid = list(set(dataset['graph_id']))
    seq = []
    y = []
    for gid in all_gid:
        
        y_loc = list(status['graph_id']).index(gid)
        adas = status['ADAS'][y_loc]
        if np.isnan(adas):
            continue
        y.append(adas)
        
        image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
        image_gid = image_gid[:, 0:260].astype('float32')
        num_img = image_gid.shape[0]
        if num_img < 9:
            sup_img = np.zeros((9 - num_img, 260)) - 1
            image_gid = np.vstack([image_gid, sup_img])
    
        seq.append(list(image_gid))
        
    return np.array(seq), np.array(y)

for num_cv in range(5):
    
    j = num_cv + 1
    train_seq, train_y = convert_seq_adas(j, 'train')
    test_seq, test_y = convert_seq_adas(j, 'test')
    
    for cv_run in range(1,6):
       
        dif12_rnn = adas_rnn(0.001, 0.05, 259)
        dif12_fit = dif12_rnn.fit([train_seq[:,:,0:259]], train_y,
                                  batch_size = 32, epochs = 100, 
                                  validation_data = ([test_seq[:,:,0:259]], test_y))
        
        # dif12_noimg_rnn = adas_rnn(0.001, 0.05, 3)
        # dif12_noimg_fit = dif12_noimg_rnn.fit([train_seq[:,:,256:259]], train_y,
        #                           batch_size = 32, epochs = 100, 
        #                           validation_data = ([test_seq[:,:,256:259]], test_y))
        
        dif12_img_rnn = adas_rnn(0.001, 0.05, 256)
        dif12_img_fit = dif12_img_rnn.fit([train_seq[:,:,0:256]], train_y,
                                  batch_size = 32, epochs = 100, 
                                  validation_data = ([test_seq[:,:,0:256]], test_y))
        
        hist_df = pd.DataFrame(dif12_fit.history)
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/rnn/adas_results.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        # hist_df = pd.DataFrame(dif12_noimg_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/rnn/adas_noimg_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)
            
        hist_df = pd.DataFrame(dif12_img_fit.history)
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j),'/run', str(cv_run),  '/rnn/adas_img_results.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)