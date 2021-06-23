#one CNN for feature extraction
import tensorflow as tf 
import numpy as np
import pandas as pd
import nibabel as nib
import random
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.activations import relu
from sklearn.model_selection import StratifiedKFold
import logging


bp_images, bp_diags, bp_scores, bp_diags_y, bp_scores_y, visits_info = get_data('BP')

lt_images, lt_diags, lt_scores, lt_diags_y, lt_scores_y, _ = get_data('LT')

#split train and test according number of samples (not number of images)
all_gid = list(bp_images.keys())
outcomes = list(bp_diags_y.values())

n = len(all_gid)

#create 5-fold CV (index) (do not run this for a second time)
num_folds = 5
# i = 0
# kfold = StratifiedKFold(n_splits=num_folds, shuffle = True)

# #save training gid and test gid
# for t1, t2 in kfold.split(np.zeros(n), outcomes):
    
#     i = i+1
#     train_indexes = pd.DataFrame({'train': t1})
#     test_indexes = pd.DataFrame({'test': t2})
    
#     hist_csv_file = ''.join(['Code/Info/graphDIF12/cv',str(i), '/train.csv'])
#     with open(hist_csv_file, mode='w') as f:
#         train_indexes.to_csv(f)
        
#     hist_csv_file = ''.join(['Code/Info/graphDIF12/cv',str(i), '/test.csv'])
#     with open(hist_csv_file, mode='w') as f:
#         test_indexes.to_csv(f)
    
#run this for different cv by changing the value of i
#i should be change through 1 to 5

#for each cv, run 5 times
for cv_run in range(1, 6):
    
    for j in range(5):
        
        num_cv = j + 1
        train_index = pd.read_csv(''.join(['Code/Info/graphDIF12/cv',str(num_cv), '/train.csv']))
        test_index = pd.read_csv(''.join(['Code/Info/graphDIF12/cv',str(num_cv), '/test.csv']))
        
        train_index = list(train_index['x'])
        test_index = list(test_index['x'])
         
        bp_train_images = []
        bp_test_images = []
        
        lt_train_images = []
        lt_test_images = []
        
        train_diags = []
        test_diags = []
        
        train_scores = []
        test_scores = []
        
        
        for gid in train_index:        

            bp_train_images = bp_train_images + list(bp_images[gid])
            lt_train_images = lt_train_images + list(lt_images[gid])
            train_diags = train_diags + list(bp_diags[gid])
            train_scores = train_scores + list(bp_scores[gid])
        
        for gid in test_index:

            bp_test_images = bp_test_images + list(bp_images[gid])
            lt_test_images = lt_test_images + list(lt_images[gid])
            test_diags = test_diags + list(bp_diags[gid])
            test_scores = test_scores + list(bp_scores[gid])
        
        
        #to np array
        bp_train_images = np.array(bp_train_images)
        bp_test_images = np.array(bp_test_images)
        lt_train_images = np.array(lt_train_images)
        lt_test_images = np.array(lt_test_images)
        
        train_diags = np.array(train_diags)
        train_scores = np.array(train_scores)
        test_diags = np.array(test_diags)
        test_scores = np.array(test_scores)
        
        
        nan_index = list(np.argwhere(np.equal(train_diags[:,1], 1)).reshape(-1))
        seq_diags = to_categorical(np.delete(train_diags[:,0], nan_index))
        
        
        nan_test_index = list(np.argwhere(np.equal(test_diags[:,1], 1)).reshape(-1))
        seq_test_diags = to_categorical(np.delete(test_diags[:,0], nan_test_index))
        
        class CNNBlock(Layer):
            def __init__(self, input_dim = 37*33*17, name = None, alpha = 0.01):
                super(CNNBlock, self).__init__(name = name)
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
                
        
        def mymodel(input_shape = (37,33,17,1), alpha = 0.01, lr = 0.001):
            
            input1 = Input(shape = input_shape)
            block = CNNBlock(alpha = alpha, name = 'l')
            output1 = block(input1)
            dense0 = Dense(256, activation = 'relu', name = 'd1', kernel_regularizer = l2(alpha))(output1)
            dense1 = Dense(128, activation = 'relu', name = 'extract', kernel_regularizer = l2(alpha))(dense0)
            drop2 = Dropout(0.2)(dense1)
            dense2 = Dense(3, activation = 'softmax')(drop2)
            opt = Adam(lr = lr)
            cnn_model = Model([input1], dense2)
            cnn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
        
            return cnn_model
            
        bp_model = mymodel(bp_train_images.shape[1:5], 0.02, 0.0005)
        lt_model = mymodel(lt_train_images.shape[1:5], 0.02, 0.0005)
        
        #don't use images whose outcome/score are unavailable
        bp_train = bp_model.fit([np.delete(bp_train_images, nan_index, axis = 0)], seq_diags, batch_size = 40, epochs = 50,
         shuffle = True, validation_data = ([np.delete(bp_test_images, nan_test_index, axis = 0)], seq_test_diags), verbose = 2)
        
        lt_train = lt_model.fit([np.delete(lt_train_images, nan_index, axis = 0)], seq_diags, batch_size = 40, epochs = 50,
         shuffle = True, validation_data = ([np.delete(lt_test_images, nan_test_index, axis = 0)], seq_test_diags), verbose = 2)
        
        #save extracted features as dict with graph_id's as keys
        bp_train_ex = dict()
        bp_test_ex = dict()
        lt_train_ex = dict()
        lt_test_ex = dict()
        
        train_ex = dict()
        test_ex = dict()
        
        #specify the layer from which the output is from
        bp_extract_layer = Model(inputs = bp_model.inputs,
                outputs = bp_model.get_layer('extract').output)
        lt_extract_layer = Model(inputs = lt_model.inputs,
                outputs = lt_model.get_layer('extract').output)
        
        train_diags_y = []
        test_diags_y = []
        
        train_scores_y = []
        test_scores_y = []
        
        for gid in train_index:   

            train_diags_y.append(bp_diags_y[gid])
            train_scores_y.append(bp_scores_y[gid])
            bp_ex = bp_extract_layer(bp_images[gid])
            lt_ex = lt_extract_layer(lt_images[gid])
            cur_scores = bp_scores[gid]
            cur_diag = bp_diags[gid][:,0].reshape((-1,1))
            bp_train_ex[gid] = bp_ex
            lt_train_ex[gid] = lt_ex
            train_ex[gid] = np.concatenate([bp_ex, lt_ex, cur_scores, cur_diag], axis = 1)
            
        for gid in test_index:   

            test_diags_y.append(bp_diags_y[gid])
            test_scores_y.append(bp_scores_y[gid])
            bp_ex = bp_extract_layer(bp_images[gid])
            lt_ex = lt_extract_layer(lt_images[gid])
            cur_scores = bp_scores[gid]
            cur_diag = bp_diags[gid][:,0].reshape((-1,1))
            bp_test_ex[gid] = bp_ex
            lt_test_ex[gid] = lt_ex
            test_ex[gid] = np.concatenate([bp_ex, lt_ex, cur_scores, cur_diag], axis = 1)
            
        #np array
        train_diags_y = np.array(train_diags_y)
        test_diags_y = np.array(test_diags_y)
        train_scores_y = np.array(train_scores_y)
        test_scores_y = np.array(test_scores_y)
        
        #save as dataframe from dictionary
        #define column names
        col_names = []
        for num_feats in range(128*2):
            col_names.append(num_feats+1)
        
        col_names = col_names + ['ADAS', 'MMSE', 'CDR', 'DXCURREN']
        df_train = pd.DataFrame.from_dict({(k,i):vs[i] for k,vs in train_ex.items()
                                     for i in range(len(vs))}, orient = 'index', columns=col_names)
        
        df_test = pd.DataFrame.from_dict({(k,i):vs[i] for k,vs in test_ex.items()
                                     for i in range(len(vs))}, orient = 'index', columns=col_names)
            
        
        df_train_gid = [index[0] for index in df_train.index]
        df_test_gid = [index[0] for index in df_test.index]
        
        df_train['graph_id'] = df_train_gid
        df_test['graph_id'] = df_test_gid
        
        #the saved extracted features (node features) include extracted LT and BC,
        #as well as cognitive score (mmse, adas, cdr) and cognitive status
        #and graph_id (so we know which subjects are used as training/test sets)
        train_path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/train_nodes.csv'])
        test_path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/test_nodes.csv'])
        df_train.to_csv(train_path, index = False)
        df_test.to_csv(test_path, index = False)
        

