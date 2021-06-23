#CNN for prediction (last available PET image)
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
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.activations import relu



class CNNBlock(Layer):
    def __init__(self, input_dim = 30*30*30, name = None, alpha = 0.01):
        super(CNNBlock, self).__init__(name = name)
        self.l1 = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer = l2(alpha))
        self.l2 = MaxPooling3D(pool_size=(2, 2, 2))
        self.l3 = BN()
        self.l4 = Dropout(0.4)
        self.l5 = Conv3D(16, kernel_size=(3,3,3), activation = 'relu', kernel_regularizer = l2(alpha))
        self.l6 = MaxPooling3D(pool_size = (2,2,2))
        self.l7 = BN()
        self.l8 = Dropout(0.4)
        self.l9 = Conv3D(32, kernel_size=(3,3,3), activation = 'relu', kernel_regularizer = l2(alpha))
        self.l10 = MaxPooling3D(pool_size = (2,2,2))
        self.l11 = BN()
        self.l12 = Dropout(0.4)
        self.l13 = Flatten()

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)

        return x

def simmodel(input_shape = (30,30,30,1), alpha = 0.01, lr = 0.001):
    
    input1 = Input(shape = input_shape)
    block = CNNBlock(alpha = alpha, name = 'l')
    output1 = block(input1)
    dense0 = Dense(256, activation = 'relu', name = 'd1', kernel_regularizer = l2(alpha))(output1)
    dense1 = Dense(128, activation = 'relu', name = 'extract', kernel_regularizer = l2(alpha))(dense0)
    drop2 = Dropout(0.4)(dense1)
    dense2 = Dense(2, activation = 'softmax')(drop2)
    opt = Adam(lr = lr)
    cnn_model = Model([input1], dense2)
    cnn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return cnn_model

#get the last image of each subject


def get_last(data_X, data_y):
    
    last_img = []
    last_y = []
    
    for gid in data_X.keys():
        
        cur_imgs = data_X[gid]
        cur_y = data_y[gid]
        n_img = cur_imgs.shape[0]
        cur_img = cur_imgs[n_img - 1]
        last_img.append(cur_img)
        last_y.append(cur_y)
        
    return np.array(last_img), np.array(last_y)

#sim2
train2_lastX, train2_lasty = get_last(train2_X, train2_final_diag)
test2_lastX, test2_lasty = get_last(test2_X, test2_final_diag)
    
cnn2 = simmodel(train2_lastX.shape[1:5], alpha = 0.2, lr = 0.001)
cnn2_train = cnn2.fit([train2_lastX], to_categorical(train2_lasty),
                      validation_data = ([test2_lastX], to_categorical(test2_lasty)),
                      batch_size = 40, epochs = 50, shuffle = True)

#sim3
train3_lastX, train3_lasty = get_last(train3_X, train3_final_diag)
test3_lastX, test3_lasty = get_last(test3_X, test3_final_diag)
    
cnn3 = simmodel(train3_lastX.shape[1:5], alpha = 0.2, lr = 0.001)
cnn3_train = cnn3.fit([train3_lastX], to_categorical(train3_lasty),
                      validation_data = ([test3_lastX], to_categorical(test3_lasty)),
                      batch_size = 40, epochs = 50, shuffle = True)
    
