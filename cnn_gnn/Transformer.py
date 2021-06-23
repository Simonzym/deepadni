#build Transformer for extracted features
import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import nibabel as nib
import random
import pickle
from itertools import chain
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

#outcome
def brier(y_true, y_pred):
    return np.mean(np.sum(np.power(y_true - y_pred, 2), axis = 1))


class TransformerBlock(Layer):
    def __init__(self, feats_dim, num_heads, ff_dim, rate=0.1, alpha = 0.01):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feats_dim,
                                             kernel_regularizer = l2(alpha))
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu", kernel_regularizer = l2(alpha)), 
             Dense(feats_dim, kernel_regularizer = l2(alpha)),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        # self.query = Dense(9, activation = 'linear')
        # self.key = Dense(9, activation = 'linear')
        # self.value = Dense(9, activation = 'linear')

    def call(self, inputs, training):
        # query = tf.transpose(self.query(tf.transpose(inputs, perm = [0,2,1])), perm = [0,2,1])
        # key = tf.transpose(self.key(tf.transpose(inputs, perm = [0,2,1])), perm = [0,2,1])
        # value = tf.transpose(self.value(tf.transpose(inputs, perm = [0,2,1])), perm = [0,2,1])
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
def build_trans(inputs_shape, ffn_dim, alpha = 0.01, lr = 0.001, rate = 0.1):
    
    inputs = Input(shape = inputs_shape)
    trans_block1 = TransformerBlock(inputs_shape[1], 5, ffn_dim, alpha = alpha, rate = rate)
    trans_block2 = TransformerBlock(inputs_shape[1], 5, ffn_dim, alpha = alpha, rate = rate)
    #trans_block3 = TransformerBlock(inputs_shape[1], 3, ffn_dim, alpha = alpha, rate = rate)
    #trans_block4 = TransformerBlock(inputs_shape[1], 3, ffn_dim, alpha = alpha, rate = rate)
    x = trans_block1(inputs)
    x = trans_block2(x)
    #x = trans_block3(x)
    #x = trans_block4(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = Dropout(rate)(x)
    x = Dense(128, activation = 'relu', kernel_regularizer = l2(alpha))(x)
    x = Dropout(rate)(x)
    outputs = Dense(3, activation = 'softmax')(x)
    
    trans_model = Model(inputs = inputs, outputs = outputs)
    opt = Adam(lr = lr)
    trans_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    
    return trans_model
    
_, _, _, _, _, visits_info = get_data('BP')
#take the extracted nodes for each graph, and build sequences
def trans_seq(num_cv, cv_run, set_type = 'train', start = 0, end = 259):
    
    path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/', 'run', str(cv_run), '/' ,set_type, '_nodes.csv'])
    graph_path = 'Code/Info/graphDIF12/graphs.csv'
    
    dataset = pd.read_csv(path)
    status = pd.read_csv(graph_path)
    all_gid = list(set(dataset['graph_id']))
    seq = []
    y = []
    num_seq = []
    
    d_model = end - start
    
    for gid in all_gid:
        
        times = visits_info[gid]
        image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
        image_gid = image_gid[:, start:end].astype('float32')
        num_img = image_gid.shape[0]
        num_seq.append(num_img)
        #calculate pe (positional embedding) for each sequence       
        pe = np.zeros((num_img, d_model))
        position = np.expand_dims(times/6, axis = 1)
        div_term_1 = np.exp(np.arange(0, d_model, 2) * -np.log(10000)/d_model)
        div_term_2 = np.exp((np.arange(1, d_model, 2)-1) * -np.log(10000)/d_model)
        pe[:, 0::2] = np.sin(position * div_term_1)
        pe[:, 1::2] = np.cos(position * div_term_2)
        #features + positional embedding
        image_pe_gid = image_gid + pe
        #exp
        if num_img < 9:
            sup_img = np.zeros((9 - num_img, d_model))
            image_pe_gid = np.vstack([image_pe_gid, sup_img])
               
        y_loc = list(status['graph_id']).index(gid)
        y.append(status['label'][y_loc])
        seq.append(list(image_pe_gid))
     
    print(9 in num_seq)    
    return np.array(seq), np.array(y)
    
for cv_run in range(1,6):
    for num_cv in range(5):
    
        j = num_cv + 1

        
        train_seq, train_y = trans_seq(j, cv_run, 'train')
        test_seq, test_y = trans_seq(j, cv_run, 'test')
        
        
        dif12_trans = build_trans(train_seq[0].shape, 256, alpha = 0.005, lr = 0.001, rate = 0.1)

        metric = []
        img_metirc = []
        for ep in range(1, 101):
            
            dif12_fit = dif12_trans.fit([train_seq], to_categorical(train_y),
                                      batch_size = 32, epochs = 1,  shuffle = True,
                                      validation_data = ([test_seq], to_categorical(test_y)), verbose = 2)     
            preds = dif12_trans.predict(test_seq)
            brier_score = brier(to_categorical(test_y), preds)
            auc = roc_auc_score(to_categorical(test_y), preds, multi_class = 'ovo')
            history = list(dif12_fit.history.values())
            history = list(chain(*history))
            history.append(brier_score)
            history.append(auc)
            metric.append(history)
        # train_noimg_seq, train_noimg_y = trans_seq(j, 'train', 256, 259)
        # test_noimg_seq, test_noimg_y = trans_seq(j, 'test', 256, 259)
        
        # train_img_seq, train_img_y = trans_seq(j, cv_run, 'train', 0, 256)
        # test_img_seq, test_img_y = trans_seq(j, cv_run, 'test', 0, 256)
        

        # dif12_noimg_trans = build_trans(train_noimg_seq[0].shape, 256, alpha = 0.005, lr = 0.001, rate = 0.1)
        # dif12_noimg_fit = dif12_noimg_trans.fit([train_noimg_seq], to_categorical(train_noimg_y),
        #                           batch_size = 32, epochs = 100, shuffle = True,
        #                           validation_data = ([test_noimg_seq], to_categorical(test_noimg_y)))
        
        # dif12_img_trans = build_trans(train_img_seq[0].shape, 256, alpha = 0.005, lr = 0.001, rate = 0.1)
        # dif12_img_fit = dif12_img_trans.fit([train_img_seq], to_categorical(train_img_y),
        #                           batch_size = 32, epochs = 100, shuffle = True,
        #                           validation_data = ([test_img_seq], to_categorical(test_img_y)), verbose = 2)
        
        hist_df = pd.DataFrame(metric, columns = ['loss', 'Accuracy', 'val_loss', 'val_accuracy', 'brier', 'auc'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/trans/results.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        # hist_df = pd.DataFrame(dif12_noimg_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/trans/noimg_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)
            
        # hist_df = pd.DataFrame(dif12_img_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/trans/img_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)