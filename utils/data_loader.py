import numpy as np 
from scipy.sparse import load_npz
import random
import sys
import time 

class DataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.rating_fname = data_dir + 'rating.npz'
        self.tr_m_fname = data_dir + 'train_mask.npz'
        self.v_m_fname = data_dir + 'val_mask.npz'
        self.current_X_tr_ind = 0
        self.current_Y_tr_ind = 0
        self.current_X_val_ind = 0
        self.current_Y_val_ind = 0

    def load_data(self):
        self.R = load_npz(self.rating_fname)
        self.N, self.M = self.R.shape
        print('Original data: %d x %d' %(self.R.shape[0], self.R.shape[1]))
        val_set = np.unique(self.R.data)
        self.min_val = float(val_set[0]) 
        self.max_val = float(val_set[-1])
        self.train_mask = load_npz(self.tr_m_fname).astype(np.float32)
        self.val_mask = load_npz(self.v_m_fname).astype(np.float32)
        print('Finished loading data')
        self.X_tr_indices = np.arange(self.N)
        self.Y_tr_indices = np.arange(self.M)
        self.X_val_indices = np.arange(self.N)
        self.Y_val_indices = np.arange(self.M)
        print('Finished initializing indices')

    def split(self):
        self.R_tr = self.R.multiply(self.train_mask)  
        self.R_val = self.R.multiply(self.val_mask)
        self.X_tr = self.R_tr.copy()
        self.Y_tr = self.R_tr.copy().T.tocsr()

    def shuffle_indices(self, for_x=False, for_y=False):
        if for_x:
            print('Shuffle train X indices')
            self.X_tr_indices = np.random.permutation(xrange(self.N))
        if for_y:
            print('Shuffle train Y indices')
            self.Y_tr_indices = np.random.permutation(xrange(self.M))

    def get_X_dim(self):
        return self.X_tr.shape[1]

    def get_Y_dim(self):
        return self.Y_tr.shape[1]

    def get_toread_indices(self, current, all_indices, batch_size):
        ''' Get indices of samples to-be-read from all_indices
        '''
        start = current
        end = current + batch_size
        n_samples = all_indices.shape[0]
        to_read = 0
        flag = False
        if end > n_samples:
            to_read = end - n_samples
            end = n_samples
            flag = True
        to_read_indices = all_indices[start:end]
        start = end   
        if to_read > 0:
            to_read_indices = np.append(to_read_indices, all_indices[0:to_read])     
            start = 0
        return to_read_indices, start, flag

    def get_elements(self, R, x_indices, y_indices):
        ''' Get elements from R corresponding to the rows and columns indices 
        '''
        n = x_indices.shape[0]
        m = y_indices.shape[0]
        values = np.zeros((n,m)).astype(np.float32)
        value_ind1, value_ind2 = np.meshgrid(x_indices, y_indices)
        ind1, ind2 = np.meshgrid(xrange(n), xrange(m))
        values[ind1.flatten(),ind2.flatten()] = R[value_ind1.flatten(), value_ind2.flatten()]
        return values

    def next_batch(self, bs_x, bs_y, dataset):
        ''' read the next training batch 
        '''
        if dataset == 'train':
            start_x = self.current_X_tr_ind
            start_y = self.current_Y_tr_ind
            full_mask = self.train_mask
            full_R = self.R_tr
            all_x_indices = self.X_tr_indices
            all_y_indices = self.Y_tr_indices
        elif dataset == 'val':
            start_x = self.current_X_val_ind
            start_y = self.current_Y_val_ind
            full_mask = self.val_mask
            full_R = self.R_val
            all_x_indices = self.X_val_indices
            all_y_indices = self.Y_val_indices

        x_indices, start_x, flag_x = self.get_toread_indices(start_x, all_x_indices, bs_x)
        y_indices, start_y, flag_y = self.get_toread_indices(start_y, all_y_indices, bs_y)

        x = self.X_tr[x_indices,:].todense()
        y = self.Y_tr[y_indices,:].todense()

        R = self.get_elements(full_R, x_indices, y_indices)
        mask = self.get_elements(full_mask, x_indices, y_indices)
        
        if dataset == 'train':
            self.current_X_tr_ind = start_x
            self.current_Y_tr_ind = start_y
        elif dataset == 'val':
            self.current_X_val_ind = start_x
            self.current_Y_val_ind = start_y

        if dataset == 'train':
            if flag_x:
                self.shuffle_indices(for_x=True)
            if flag_y:
                self.shuffle_indices(for_y=True)
        flag = flag_x or flag_y
        return x, y, R, mask, flag