import os
import numpy as np
from tqdm import tqdm
from utils import split_data_label, process_batch_ts
from sklearn.utils import shuffle


def load_data(root='data', name='chapman', length=None, overlap=0, norm=True, shuffle=True, task=None):
    '''
    load correspondent training, validation, and test data and labels
    '''
    if name == 'chapman':
        dir = os.path.join(root, name)
        return load_chapman(dir, length, overlap, norm, shuffle, task)
    else:
        raise ValueError(f'Unknown dataset name: {name}')
    
    
def load_chapman(root='data/chapman', length=None, overlap=0, norm=True, shuff=True, task=None):
    data_path = os.path.join(root, 'feature')
    label_path = os.path.join(root, 'label', 'label.npy')
    
    labels = np.load(label_path)
    
    pids_sb = list(labels[np.where(labels[:, 0]==0)][:, 1])
    pids_af = list(labels[np.where(labels[:, 0]==1)][:, 1])
    pids_gsvt = list(labels[np.where(labels[:, 0]==2)][:, 1])
    pids_sr = list(labels[np.where(labels[:, 0]==3)][:, 1])
    
    train_ids = pids_sb[:-500] + pids_af[:-500] + pids_gsvt[:-500] + pids_sr[:-500]
    valid_ids = pids_sb[-500:-250] + pids_af[-500:-250] + pids_gsvt[-500:-250] + pids_sr[-500:-250]
    test_ids = pids_sb[-250:] + pids_af[-250:] + pids_gsvt[-250:] + pids_sr[-250:]
    
    filenames = []
    for fn in os.listdir(data_path):
        filenames.append(fn)
    filenames.sort()
    
    train_trials = []
    train_labels = []
    valid_trials = []
    valid_labels = []
    test_trials = []
    test_labels = []
    
    for i, fn in enumerate(tqdm(filenames, desc=f'=> Loading Chapman')):
        label = labels[i]
        feature = np.load(os.path.join(data_path, fn))
        for trial in feature:
            if i+1 in train_ids:
                train_trials.append(trial)
                train_labels.append(label)
            elif i+1 in valid_ids:
                valid_trials.append(trial)
                valid_labels.append(label)
            elif i+1 in test_ids:
                test_trials.append(trial)
                test_labels.append(label)
                
    X_train = np.array(train_trials)
    X_val = np.array(valid_trials)
    X_test = np.array(test_trials)
    y_train = np.array(train_labels)
    y_val = np.array(valid_labels)
    y_test = np.array(test_labels)
    
    if shuff:
        X_train, y_train = shuffle(X_train, y_train)
        X_val, y_val = shuffle(X_val, y_val)
        X_test, y_test = shuffle(X_test, y_test)
    
    if norm:
        X_train = process_batch_ts(X_train, normalized=True, bandpass_filter=False)
        X_val = process_batch_ts(X_val, normalized=True, bandpass_filter=False)
        X_test = process_batch_ts(X_test, normalized=True, bandpass_filter=False)
      
    if length:
        # X_train, y_train = segment(X_train, y_train, split)
        # X_val, y_val = segment(X_val, y_val, split)
        # X_test, y_test = segment(X_test, y_test, split)
        
        X_train, y_train = split_data_label(X_train,y_train, sample_timestamps=length, overlapping=overlap)
        # no neighbor contrast in validation and test set
        X_val, y_val = split_data_label(X_val,y_val, sample_timestamps=length//2, overlapping=overlap)
        X_test, y_test = split_data_label(X_test,y_test, sample_timestamps=length//2, overlapping=overlap)
        
    if task == 'cmsc':
        X_train, y_train = cmsc_split(X_train, y_train)
        
    elif task == 'cmsmlc':
        X_train, y_train = cmsmlc_split(X_train, y_train) 
        
    elif task == 'cmlc':
        X_train, y_train = cmlc_split(X_train, y_train)   
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def cmsc_split(x, y):
    length = x.shape[1]
    nleads = x.shape[-1]
    assert length % 2 == 0
    
    x = x.transpose(2, 0, 1).reshape(-1, 2, int(length/2), 1)
    y = np.tile(y, (nleads, 1))
    
    return x, y


def cmsmlc_split(x, y):
    length = x.shape[1]
    batch_size = x.shape[0]
    assert length % 2 == 0
    
    x = x.transpose(0, 2, 1).reshape(batch_size, -1, int(length/2), 1)
    
    return x, y


def cmlc_split(x, y):
    length = x.shape[1]
    batch_size = x.shape[0]
    x = x.transpose(0, 2, 1).reshape(batch_size, -1, length, 1)
    
    return x, y
