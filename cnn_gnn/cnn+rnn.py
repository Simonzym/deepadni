#CNN+RNN (dynamic)
#read train gid and test gid
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

bp_images, bp_diags, bp_scores, bp_diags_y, bp_scores_y = get_data('BP')

lt_images, lt_diags, lt_scores, lt_diags_y, lt_scores_y = get_data('LT')

train_gid = pd.read_csv('Code/Info/graphDIF12/train_gid.csv')
test_gid = pd.read_csv('Code/Info/graphDIF12/test_gid.csv')

#get data ready
bp_train_stack = []
lt_train_stack = []

bp_test_stack = []
lt_test_stack = []

train_seq = []
test_seq = []

train_y = []
test_y = []

for gid in train_gid['graph_id']:
    
    cur_bp = bp_images[gid]
    cur_lt = lt_images[gid]
    cur_scores = bp_scores[gid]
    cur_diag = bp_diags[gid].reshape(-1, 1)
    cur_seq = np.hstack((cur_scores, cur_diag))
    cur_y = bp_diags_y[gid]
    n_imgs = cur_bp.shape[0]
    add_bp = np.zeros((9 - n_imgs, 37, 33, 17, 1))
    add_lt = np.zeros((9 - n_imgs, 30, 27, 16, 1))
    
    bp_train_stack.append(np.vstack((cur_bp, add_bp)))
    lt_train_stack.append(np.vstack((cur_lt, add_lt)))
    
    add_seq = np.zeros((9 - n_imgs, 4)) - 1
    train_seq.append(np.vstack((cur_seq, add_seq)))
    
    train_y.append(cur_y)

for gid in test_gid['graph_id']:
    
    cur_bp = bp_images[gid]
    cur_lt = lt_images[gid]
    cur_scores = bp_scores[gid]
    cur_diag = bp_diags[gid].reshape(-1, 1)
    cur_seq = np.hstack((cur_scores, cur_diag))
    cur_y = bp_diags_y[gid]
    n_imgs = cur_bp.shape[0]
    add_bp = np.zeros((9 - n_imgs, 37, 33, 17, 1))
    add_lt = np.zeros((9 - n_imgs, 30, 27, 16, 1))
    
    bp_test_stack.append(np.vstack((cur_bp, add_bp)))
    lt_test_stack.append(np.vstack((cur_lt, add_lt)))
    
    add_seq = np.zeros((9 - n_imgs, 4)) - 1
    test_seq.append(np.vstack((cur_seq, add_seq)))
    
    test_y.append(cur_y)

bp_train_stack = np.array(bp_train_stack)
bp_test_stack = np.array(bp_test_stack)

lt_train_stack = np.array(lt_train_stack)
lt_test_stack = np.array(lt_test_stack)

train_seq = np.array(train_seq)
test_seq = np.array(test_seq)

train_y = np.array(train_y)
test_y = np.array(test_y)
    
class CRBlock(Layer):
    def __init__(self, input_dim = 37*33*17, name = None, alpha = 0.01):
        super(CRBlock, self).__init__(name = name)
        self.l1 = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer = l2(alpha), padding = 'same')
        self.l2 = MaxPooling3D(pool_size=(2, 2, 2), padding = 'same')
        self.l3 = BN()
        self.l4 = Dropout(0.4)
        self.l5 = Conv3D(16, kernel_size=(3,3,3), activation = 'relu', kernel_regularizer = l2(alpha), padding = 'same')
        self.l6 = MaxPooling3D(pool_size = (2,2,2), padding = 'same')
        self.l7 = BN()
        self.l8 = Dropout(0.4)
        self.l9 = Conv3D(32, kernel_size=(3,3,3), activation = 'relu', kernel_regularizer = l2(alpha), padding = 'same')
        self.l10 = MaxPooling3D(pool_size = (2,2,2), padding = 'same')
        self.l11 = BN()
        self.l12 = Dropout(0.4)
        self.l13 = Flatten()
        self.l14 = Dense(256, activation = 'relu', name = 'd1', kernel_regularizer = l2(alpha))
        self.l15 = Dense(128, activation = 'relu', name = 'extract', kernel_regularizer = l2(alpha))
        self.nontrain_w = tf.Variable(initial_value=tf.zeros([input_dim, 128]),
                                        trainable = False, dtype = tf.float32)

    def call(self, inputs):
        
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        #x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        x = self.l14(x)
        x = self.l15(x)
        
        x1 = Flatten()(inputs)
        x1 = tf.matmul(x1, self.nontrain_w) - 1
            
        #if inputs indicate missing values, don't train
        outputs = keras.backend.switch(tf.reduce_all(tf.equal(inputs, 0)), x1, x)

        return outputs

bp_input = Input(shape = (9, 37, 33, 17, 1))
bp_block = CRBlock(name = 'bp')
bp_output1 = bp_block(bp_input[:,0,:,:,:])
bp_output2 = bp_block(bp_input[:,1,:,:,:])
bp_output3 = bp_block(bp_input[:,2,:,:,:])
bp_output4 = bp_block(bp_input[:,3,:,:,:])
bp_output5 = bp_block(bp_input[:,4,:,:,:])
bp_output6 = bp_block(bp_input[:,5,:,:,:])
bp_output7 = bp_block(bp_input[:,6,:,:,:])
bp_output8 = bp_block(bp_input[:,7,:,:,:])
bp_output9 = bp_block(bp_input[:,8,:,:,:])
bp_merged = Concatenate()([bp_output1, bp_output2, bp_output3, bp_output4,
                           bp_output5, bp_output6, bp_output7, bp_output8,
                           bp_output9])
bp_shape = Reshape((9, 128))(bp_merged)

lt_input = Input(shape = (9, 30, 27, 16, 1))
lt_block = CRBlock(30*27*16, name = 'lt')
lt_output1 = lt_block(lt_input[:,0,:,:,:])
lt_output2 = lt_block(lt_input[:,1,:,:,:])
lt_output3 = lt_block(lt_input[:,2,:,:,:])
lt_output4 = lt_block(lt_input[:,3,:,:,:])
lt_output5 = lt_block(lt_input[:,4,:,:,:])
lt_output6 = lt_block(lt_input[:,5,:,:,:])
lt_output7 = lt_block(lt_input[:,6,:,:,:])
lt_output8 = lt_block(lt_input[:,7,:,:,:])
lt_output9 = lt_block(lt_input[:,8,:,:,:])
lt_merged = Concatenate()([lt_output1, lt_output2, lt_output3, lt_output4,
                           lt_output5, lt_output6, lt_output7, lt_output8,
                           lt_output9])
lt_shape = Reshape((9, 128))(lt_merged)

seq_input = Input(shape = (9, 4))
rnn_input = Concatenate()([bp_shape, lt_shape])
mask = Masking(mask_value = -1)(rnn_input)
rnn1 = Bidirectional(GRU(256, activation = 'relu', return_sequences=True, kernel_regularizer = l2(0.01)))(mask)
drop1 = Dropout(0.4)(rnn1)
rnn2 = Bidirectional(GRU(128, activation = 'relu', kernel_regularizer = l2(0.01)))(drop1)
dense1 = Dense(256, activation = 'relu', kernel_regularizer = l2(0.01))(rnn2)
drop2 = Dropout(0.4)(dense1)
dense2 = Dense(3, activation = 'softmax', kernel_regularizer = l2(0.01))(drop2)
opt = Adam(lr = 0.001)

cr_model = Model([bp_input, lt_input, seq_input], dense2)
cr_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

cr_result = cr_model.fit([bp_train_stack, lt_train_stack, train_seq], to_categorical(train_y),
                         validation_data = ([bp_test_stack, lt_test_stack, test_seq], to_categorical(test_y)),
                         batch_size = 16, epochs = 50)