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
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

def brier(y_true, y_pred):
    return K.mean(K.sum(K.pow(y_true - y_pred, 2), axis = 1))


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
    outputs = Dense(2, activation = 'softmax')(x)
    
    trans_model = Model(inputs = inputs, outputs = outputs)
    opt = Adam(lr = lr)
    trans_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', tf.keras.metrics.AUC(), brier])
    
    return trans_model
    

#take the extracted nodes for each graph, and build sequences
def trans_seq(num_sim, sim_turn, set_type = 'train'):
    
    sim_run = ''.join(['graphSim', str(num_sim)])
    
    _, _, _, _, _, visits_info = get_data(sim_run, sim_turn, set_type)

    
    path = ''.join(['Code/Info/SimGraph/graphSim', str(num_sim), '/sim', str(sim_turn), '/', set_type, '/nodes.csv'])
    graph_path = ''.join(['Code/Info/SimGraph/graphSim', str(num_sim), '/sim', str(sim_turn), '/', set_type, '/graphs.csv'])
    
    dataset = pd.read_csv(path)
    status = pd.read_csv(graph_path)
    all_gid = list(set(dataset['graph_id']))
    seq = []
    y = []
    
    d_model = 128
    
    for gid in all_gid:
        
        times = visits_info[gid]
        image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
        image_gid = image_gid[:,0:128].astype('float32')
        num_img = image_gid.shape[0]
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
        
    return np.array(seq), np.array(y)
    
sims = [2, 3, 4, 7]
for num_sim in sims:
    for sim_turn in range(1, 101):
    

        train_seq, train_y = trans_seq(num_sim, sim_turn, 'train')
        test_seq, test_y = trans_seq(num_sim, sim_turn, 'test')
        
        
        sim_trans = build_trans(train_seq[0].shape, 128, alpha = 0, lr = 0.0001, rate = 0)
        sim_fit = sim_trans.fit([train_seq], to_categorical(train_y),
                                  batch_size = 32, epochs = 100,  shuffle = True,
                                  validation_data = ([test_seq], to_categorical(test_y)), verbose = 3)
        
        hist_df = pd.DataFrame(sim_fit.history)
        hist_csv_file = ''.join(['Code/Info/SimGraph/graphSim', str(num_sim), '/sim', str(sim_turn), '/trans_results.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        
