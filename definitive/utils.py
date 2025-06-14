import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



class PrepareData(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y # vedere

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


def create_data_reg(train_size):
    np.random.seed(0)
    xs = np.random.uniform(low=0., high=0.6, size=train_size)
    
    eps = np.random.normal(loc=0., scale=0.02, size=[train_size])

    ys = xs + 0.3 * np.sin(2*np.pi * (xs + eps)) + 0.3 * np.sin(4*np.pi * (xs + eps)) + eps

    xs = torch.from_numpy(xs).reshape(-1,1).float()
    ys = torch.from_numpy(ys).reshape(-1,1).float()

    return xs, ys


def create_regression_plot(X_test, y_test, train_ds):
    fig = plt.figure(figsize=(9, 6))
    plt.plot(X_test, np.median(y_test, axis=0), label='Median Posterior Predictive')
    
    # Range
    plt.fill_between(
        X_test.reshape(-1), 
        np.percentile(y_test, 0, axis=0), 
        np.percentile(y_test, 100, axis=0), 
        alpha = 0.2, color='orange', label='Range') #color='blue',
    
    # interquartile range
    plt.fill_between(
        X_test.reshape(-1), 
        np.percentile(y_test, 25, axis=0), 
        np.percentile(y_test, 75, axis=0), 
        alpha = 0.4,  label='Interquartile Range') #color='red',
    
    plt.scatter(train_ds.dataset.X, train_ds.dataset.y, label='Training data', marker='x', alpha=0.5, color='k', s=2)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylim([-1.5, 1.5])
    plt.xlim([-0.6, 1.4])

def read_data_rl(data_dir):
    '''
    Read in data for contextual bandits
    - transform context and label to one-hot encoded vectors
    '''
    df = pd.read_csv(data_dir, sep=',', header=None)
    df.columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment',
         'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root',
         'stalk-surf-above-ring','stalk-surf-below-ring','stalk-color-above-ring','stalk-color-below-ring',
         'veil-type','veil-color','ring-number','ring-type','spore-color','population','habitat']
    X = pd.DataFrame(df, columns=df.columns[1:len(df.columns)], index=df.index)
    Y = df['class']

    # transform to one-hot encoding
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(Y)
    Y_ = label_encoder.transform(Y)
    X_ = X.copy()
    for feature in X.columns:
        label_encoder.fit(X[feature])
        X_[feature] = label_encoder.transform(X[feature])

    oh_encoder = preprocessing.OneHotEncoder()
    oh_encoder.fit(X_)
    X_ = oh_encoder.transform(X_).toarray()

    return X_, Y_



   