import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import sys

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)
    
def split_data(data, seed): # info will be [ids, y]
    cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed)
    unique_labels = np.unique(data['y'])
    if len(unique_labels) > 5: # for regression
        kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        y_binned = kbins.fit_transform(data['y'].values.reshape(-1,1)).reshape(-1)
        splits = cv.split(data, y_binned)
    else: splits = cv.split(data, data['y'])
    return splits
    
def get_metrics(labels, probs, out_size):
    
    preds = np.asarray([1 if prob >= 0.5 else 0 for prob in probs])
    if out_size > 1:
        preds = preds.reshape(int(preds.shape[0]/out_size),out_size)
        probs = probs.reshape(int(probs.shape[0]/out_size),out_size)
        labels = labels.reshape(int(labels.shape[0]/out_size),out_size)
        
        bacc = balanced_accuracy_score(np.argmax(labels, axis=1), np.argmax(preds, axis=1))
        auroc = roc_auc_score(labels, probs, multi_class='ovr')
    else:
        auroc = roc_auc_score(labels, probs)
        bacc = balanced_accuracy_score(labels, preds)
    
    return auroc, bacc
