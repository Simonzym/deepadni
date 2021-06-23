#one CNN for feature extraction
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
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.activations import relu



class CNNBlock(Layer):
    def __init__(self, input_dim = 30*30*30, name = None, alpha = 0.01):
        super(CNNBlock, self).__init__(name = name)
        self.l1 = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer = l2(alpha))
        self.l2 = MaxPooling3D(pool_size=(2, 2, 2), padding = 'same')
        self.l3 = BN()
        self.l4 = Dropout(0.4)
        self.l5 = Conv3D(16, kernel_size=(3,3,3), activation = 'relu', kernel_regularizer = l2(alpha))
        self.l6 = MaxPooling3D(pool_size = (2,2,2), padding = 'same')
        self.l7 = BN()
        self.l8 = Dropout(0.4)
        self.l9 = Conv3D(32, kernel_size=(3,3,3), activation = 'relu', kernel_regularizer = l2(alpha))
        self.l10 = MaxPooling3D(pool_size = (2,2,2), padding = 'same')
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
    drop2 = Dropout(0.2)(dense1)
    dense2 = Dense(2, activation = 'softmax')(drop2)
    opt = Adam(lr = lr)
    cnn_model = Model([input1], dense2)
    cnn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return cnn_model


def get_nodes(input_model, train_image_dict, test_image_dict, folder, num_sim):
    
    train_ex = dict()
    test_ex = dict()
    
    extract_layer = Model(inputs = input_model.inputs,
            outputs = input_model.get_layer('extract').output)
    
    train_gid = list(train_image_dict.keys())
    test_gid = list(test_image_dict.keys())
    
    for gid in train_gid:   

        train_image_ex = extract_layer(train_image_dict[gid])
        train_ex[gid] = np.array(train_image_ex)
        
    for gid in test_gid:   
        
        test_image_ex = extract_layer(test_image_dict[gid])
        test_ex[gid] = np.array(test_image_ex)
        
    df_train = pd.DataFrame.from_dict({(k,i):vs[i] for k,vs in train_ex.items()
                                 for i in range(len(vs))}, orient = 'index')
    
    df_test = pd.DataFrame.from_dict({(k,i):vs[i] for k,vs in test_ex.items()
                                 for i in range(len(vs))}, orient = 'index')
        
    
    
    df_train_gid = [index[0] for index in df_train.index]
    df_test_gid = [index[0] for index in df_test.index]
    
    df_train['graph_id'] = df_train_gid
    df_test['graph_id'] = df_test_gid
    
    df_train.to_csv(''.join(['Code/Info/SimGraph/', folder, '/sim', str(num_sim), '/train/nodes.csv']), index = False)
    df_test.to_csv(''.join(['Code/Info/SimGraph/', folder, '/sim', str(num_sim), '/test/nodes.csv']), index = False)
    
#fit model for sim2
for sim_run in range(1, 101):
    
    train2_X, train2_diag, train2_final_diag, train2_images, train2_diags, _ = get_data('graphSim2', sim_run)
    test2_X, test2_diag, test2_final_diag, test2_images, test2_diags, _ = get_data('graphSim2', sim_run, 'test')
    
    sim2_model = simmodel(train2_images.shape[1:5], alpha = 0.08)
    sim2_train = sim2_model.fit([train2_images], to_categorical(train2_diags), batch_size = 40,
                                epochs = 50, shuffle = True,
                                validation_data = ([test2_images], to_categorical(test2_diags)), verbose = 2)
    
    get_nodes(sim2_model, train2_X, test2_X, 'graphSim2', sim_run)
    
        

#fit model fot sim3
for sim_run in range(1, 101):
    
    train3_X, train3_diag, train3_final_diag, train3_images, train3_diags, _ = get_data('graphSim3', sim_run)
    test3_X, test3_diag, test3_final_diag, test3_images, test3_diags, _ = get_data('graphSim3', sim_run, 'test')
    
    sim3_model = simmodel(train3_images.shape[1:5], alpha = 0.1)
    sim3_train = sim3_model.fit([train3_images], to_categorical(train3_diags), 
                            batch_size = 40, epochs = 50, shuffle = True, verbose = 2,
                            validation_data = ([test3_images], to_categorical(test3_diags)))

    get_nodes(sim3_model, train3_X, test3_X, 'graphSim3', sim_run)
    
    

#fit model for sim4
for sim_run in range(1, 101):
    
    train4_X, train4_diag, train4_final_diag, train4_images, train4_diags, _ = get_data('graphSim4', sim_run)
    test4_X, test4_diag, test4_final_diag, test4_images, test4_diags, _ = get_data('graphSim4', sim_run, 'test')
    
    sim4_model = simmodel(train4_images.shape[1:5], alpha = 0.2, lr = 0.0005)
    sim4_train = sim4_model.fit([train4_images], to_categorical(train4_diags), 
                            batch_size = 40, epochs = 50, shuffle = True, verbose = 2, 
                            validation_data = ([test4_images], to_categorical(test4_diags)))
    
    get_nodes(sim4_model, train4_X, test4_X, 'graphSim4', sim_run)
    
    
    

#fit model for sim7
for sim_run in range(1, 101):
    
    train7_X, train7_diag, train7_final_diag, train7_images, train7_diags, _ = get_data('graphSim7', sim_run)
    test7_X, test7_diag, test7_final_diag, test7_images, test7_diags, _ = get_data('graphSim7', sim_run, 'test')
    
    sim7_model = simmodel(train7_images.shape[1:5], alpha = 0.3, lr = 0.0005)
    sim7_train = sim7_model.fit([train7_images], to_categorical(train7_diags), 
                            batch_size = 40, epochs = 50, shuffle = True, verbose = 2, 
                            validation_data = ([test7_images], to_categorical(test7_diags)))

    get_nodes(sim7_model, train7_X, test7_X, 'graphSim7', sim_run)
    
    
def get_nodes(input_model, train_image_dict, test_image_dict, folder, num_sim):
    
    train_ex = dict()
    test_ex = dict()
    
    extract_layer = Model(inputs = input_model.inputs,
            outputs = input_model.get_layer('extract').output)
    
    train_gid = list(train_image_dict.keys())
    test_gid = list(test_image_dict.keys())
    
    for gid in train_gid:   

        train_image_ex = extract_layer(train_image_dict[gid])
        train_ex[gid] = np.array(train_image_ex)
        
    for gid in test_gid:   
        
        test_image_ex = extract_layer(test_image_dict[gid])
        test_ex[gid] = np.array(test_image_ex)
        
    df_train = pd.DataFrame.from_dict({(k,i):vs[i] for k,vs in train_ex.items()
                                 for i in range(len(vs))}, orient = 'index')
    
    df_test = pd.DataFrame.from_dict({(k,i):vs[i] for k,vs in test_ex.items()
                                 for i in range(len(vs))}, orient = 'index')
        
    
    
    df_train_gid = [index[0] for index in df_train.index]
    df_test_gid = [index[0] for index in df_test.index]
    
    df_train['graph_id'] = df_train_gid
    df_test['graph_id'] = df_test_gid
    
    df_train.to_csv(''.join(['Code/Info/SimGraph/', folder, '/sim', str(num_sim), '/train/nodes.csv']), index = False)
    df_test.to_csv(''.join(['Code/Info/SimGraph/', folder, '/sim', str(num_sim), '/test/nodes.csv']), index = False)
    


