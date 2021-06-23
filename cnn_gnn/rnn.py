#build RNN for extracted features
import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import nibabel as nib
import random
import pickle
import sklearn
from itertools import chain
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

#outcome
def brier(y_true, y_pred):
    return np.mean(np.sum(np.power(y_true - y_pred, 2), axis = 1))

#build model
def build_rnn(lr = 0.001, alpha = 0.02, size = 258):
    
    rnn_model = Sequential()
    rnn_model.add(Masking(mask_value=-1, input_shape = (9, size)))
    rnn_model.add(Bidirectional(GRU(256, return_sequences=True, kernel_regularizer = l2(alpha))))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Bidirectional(GRU(128, kernel_regularizer = l2(alpha))))
    rnn_model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(alpha)))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Dense(3, activation = 'softmax', kernel_regularizer = l2(alpha)))
    opt = Adam(lr = lr)
    rnn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return rnn_model


def convert_seq(num_cv, cv_run, set_type = 'train'):
    
    path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv),  '/run', str(cv_run), '/', set_type, '_nodes.csv'])
    graph_path = 'Code/Info/graphDIF12/graphs.csv'
    
    dataset = pd.read_csv(path)
    status = pd.read_csv(graph_path)
    
    all_gid = list(set(dataset['graph_id']))
    seq = []
    y = []
    for gid in all_gid:
        
        image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
        image_gid = image_gid[:, 0:260].astype('float32')
        num_img = image_gid.shape[0]
        if num_img < 9:
            sup_img = np.zeros((9 - num_img, 260)) - 1
            image_gid = np.vstack([image_gid, sup_img])
        
        y_loc = list(status['graph_id']).index(gid)
        y.append(status['label'][y_loc])
        seq.append(list(image_gid))
        
    return np.array(seq), np.array(y)

for cv_run in range(1,6):
    for num_cv in range(5):
        j = num_cv + 1
        train_seq, train_y = convert_seq(j, cv_run, 'train')
        test_seq, test_y = convert_seq(j, cv_run, 'test')
    

        
        dif12_rnn = build_rnn(0.0005, 0.05, 259)
        # dif12_img_rnn = build_rnn(0.0005, 0.05, 256)
        metric = []
        img_metirc = []
        for ep in range(1, 101):
            
            dif12_fit = dif12_rnn.fit([train_seq[:,:,0:259]], to_categorical(train_y),
                                      batch_size = 32, epochs = 1, 
                                      validation_data = ([test_seq[:,:,0:259]], to_categorical(test_y)), verbose = 2)
            preds = dif12_rnn.predict(test_seq[:,:,0:259])
            brier_score = brier(to_categorical(test_y), preds)
            auc = roc_auc_score(to_categorical(test_y), preds, multi_class = 'ovo')
            history = list(dif12_fit.history.values())
            history = list(chain(*history))
            history.append(brier_score)
            history.append(auc)
            metric.append(history)
            

        # dif12_noimg_rnn = build_rnn(0.0005, 0.05, 3)
        # dif12_noimg_fit = dif12_noimg_rnn.fit([train_seq[:,:,256:259]], to_categorical(train_y),
        #                           batch_size = 32, epochs = 100, 
        #                           validation_data = ([test_seq[:,:,256:259]], to_categorical(test_y)))
        
        # for ep in range(1, 100):
        #     dif12_img_fit = dif12_img_rnn.fit([train_seq[:,:,0:256]], to_categorical(train_y),
        #                               batch_size = 32, epochs = 1, 
        #                               validation_data = ([test_seq[:,:,0:256]], to_categorical(test_y)), verbose = 2)
            
        hist_df = pd.DataFrame(metric, columns = ['loss', 'Accuracy', 'val_loss', 'val_accuracy', 'brier', 'auc'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/rnn/diag_results.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        # hist_df = pd.DataFrame(dif12_noimg_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/rnn_noimg_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)
            
        # hist_df = pd.DataFrame(dif12_img_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/rnn/diag_img_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)