import tensorflow as tf 
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import nibabel as nib
import random

from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, concatenate
from tensorflow.keras.regularizers import l2


#get training data
def get_data(folder):

    #get path
    image_id = ''.join(['Code/Info/DIF12/', folder])

    files_id = [''.join([image_id, '/', i]) for i in os.listdir(image_id)]
    
    
    #all data for gnn
    # gnn_diag = []
    # gnn_score = []
    # gnn_image = []
    
    # all_image = []
    # all_diag = []
    # all_score = []

        
    final_diags = dict()
    final_scores = dict()


    #baseline information and final outcome
    base_out = pd.read_csv(''.join(['Code/Info/DIF12', '/base_out.csv']))
    base_gid = list(base_out['graph_id'])
    
    dyn_X = dict()
    dyn_diag = dict()
    dyn_score = dict()
    dyn_time = dict()
    for gid in base_gid:
        dyn_X[gid] = []
        dyn_diag[gid] = []
        dyn_score[gid] = []
        final_diags[gid] = []
        final_scores[gid] = []
        dyn_time[gid] = []

    for file_id in files_id:
        gid = file_id[19:-4]
        #get the location of corresponding gid in outcome file
        y_loc = base_gid.index(gid)
        #get graph level score and diagnosis
        final_diag = base_out['DXCURREN'][y_loc]
        final_score = [base_out['TOTAL11'][y_loc], base_out['MMSCORE'][y_loc], base_out['CDGLOBAL'][y_loc]]
        final_diags[gid] = np.array(final_diag)
        final_scores[gid] = np.array(final_score)
        #information of each sample
        file = pd.read_csv(file_id)
        #sequence of diagnosis and scores
        # gnn_diag_seq = []
        # gnn_score_seq = []
        # gnn_image_seq = []
        for i in range(len(file)):
            file_name = file['images'][i]
            #read image (nifti)
            image = nib.load(file_name)
            image_data = image.get_fdata().reshape((image.shape+(1,)))
            #fetch diagnosis and score
            diag = [file['DXCURREN'][i], file['DXNA'][i]]
            scores = [file['TOTAL11'][i], file['MMSCORE'][i], file['CDGLOBAL'][i]]
            time = file['time'][i]
            #append image to sample list (graph_id list)
            dyn_X[gid].append(image_data)
            dyn_diag[gid].append(diag)
            dyn_score[gid].append(scores)
            dyn_time[gid].append(time)
            #append image,diagnosis and score 
            # all_image.append(image_data)
            # all_diag.append(diag) 
            # all_score.append(score)
            
            # gnn_diag_seq.append(diag)
            # gnn_score_seq.append(score)
            # gnn_image_seq.append(image_data)
        
        #in sequence format
        # gnn_diag.append(gnn_diag_seq)
        # gnn_score.append(gnn_score_seq)
        # gnn_image.append(gnn_image_seq)

    #save in np array format
    for gid in base_gid:
        dyn_X[gid] = np.array(dyn_X[gid])
        dyn_diag[gid] = np.array(dyn_diag[gid])
        dyn_score[gid] = np.array(dyn_score[gid])
        dyn_time[gid] = np.array(dyn_time[gid])
                
    # gnn_diag = np.array(gnn_diag)
    # gnn_image = np.array(gnn_image)
    # gnn_score = np.array(gnn_score)
    # all_score = np.array(all_score)
    # all_image = np.array(all_image)
    # all_diag = np.array(all_diag)
    
    
    #final_diags = np.array(final_diags)
    #final_scores = np.array(final_scores)
        
    return dyn_X, dyn_diag, dyn_score, final_diags, final_scores, dyn_time
