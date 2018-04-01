import tensorflow as tf
import numpy as np 
import time 
import cPickle as pkl
import h5py 
import scipy.sparse
from models.dmfd import DMFD
import configs.configs_dmfd as configs

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./data/MovieLens1M/", "Data directory.")
tf.flags.DEFINE_string("snapshot_dir", "./outputs/snapshots/", "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("model_fname", "", "Name of the pretrained model checkpoints (to resume from)")
cfgs = configs.CONFIGS
DEFAULT_RATING = 3.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def string_to_array(str, dtype='int'):
    arr = str.strip().split(',')
    for i in xrange(len(arr)):
        if dtype == 'int':
            arr[i] = int(arr[i])
        elif dtype == 'float':
            arr[i] = float(arr[i])
    return arr

def embed_x(model, X, dim, bs=1000):
    n_samples = X.shape[0]
    fv = np.zeros((n_samples, dim))
    start = 0
    while True:
        end = start + bs
        if end > n_samples:
            end = n_samples
        fv[start:end,:] = model.embed_x(X[start:end,:])
        if end == n_samples:
            break
        start = end
    return fv

def embed_y(model, Y, dim, bs=1000):
    n_samples = Y.shape[0]
    fv = np.zeros((n_samples, dim))
    start = 0
    while True:
        end = start + bs
        if end > n_samples:
            end = n_samples
        fv[start:end,:] = model.embed_y(Y[start:end,:])
        if end == n_samples:
            break
        start = end
    return fv

def reconstruct_quantize_custom(latent_x, latent_y, min_val, max_val, bs, qs, Delta, Is):
    l2_norm_lx = latent_x / np.linalg.norm(latent_x, axis=1, keepdims=True)
    l2_norm_ly = latent_y / np.linalg.norm(latent_y, axis=1, keepdims=True)
    mul = np.matmul(l2_norm_lx, l2_norm_ly.T)
    mid = (max_val + min_val) / 2
    # re-scale the cosine similarity to the original entry range
    mul = mul * (mid - min_val) + mid
    k = 1e30 # a very big value
    recons = np.zeros_like(mul)
    for i in xrange(len(bs)):
        vals = Delta * sigmoid(k * (mul - bs[i])) + Is[i]
        if i < len(bs) - 1:
            mask = (mul >= qs[i]) * (mul < qs[i+1])
        else:
            mask = (mul >= qs[i]) * (mul <= qs[i+1])
        recons += vals * mask 
    return recons

def RMSE(A, B, mask):
    rmse = np.sqrt(np.sum(mask * (A - B)**2) / np.sum(mask))
    return rmse

def MAE(A, B, mask):
    mae = np.sum(mask * np.abs(A - B)) / np.sum(mask)
    return mae

def main(unused_argv):
    # load data
    R = scipy.sparse.load_npz(FLAGS.data_dir + 'rating.npz')
    val_set = np.unique(R.data)
    min_val = float(val_set[0]) 
    max_val = float(val_set[-1])

    tr_mask = scipy.sparse.load_npz(FLAGS.data_dir + 'train_mask.npz')
    val_mask = scipy.sparse.load_npz(FLAGS.data_dir + 'val_mask.npz')
    te_mask = scipy.sparse.load_npz(FLAGS.data_dir + 'test_mask.npz')
    print('Finished loading data')
    count = np.sum((tr_mask + val_mask).multiply(te_mask))
    assert count == 0, 'Train and test masks overlap !!!'

    tr_mask += val_mask
    X = R.multiply(tr_mask).todense()

    # load model 
    assert (FLAGS.snapshot_dir != "" or FLAGS.model_fname != ""), 'No pretrained model specified'
    model = DMFD(X.shape[1], X.shape[0], min_val, max_val, cfgs, phase='test', log_dir=None)

    snapshot_fname = FLAGS.model_fname if FLAGS.model_fname != "" \
        else tf.train.latest_checkpoint(FLAGS.snapshot_dir)
    model.restore(snapshot_fname)
    print('Restored from %s' %snapshot_fname)

    # complete matrix
    embed_dim = int(configs.ModelConfig.u_hidden_sizes[-1])
    lX = embed_x(model, X, embed_dim, bs=1000)
    lY = embed_y(model, X.T, embed_dim, bs=1000)
   
    bs = model.get_boundaries()
    print('Learned boundaries: ', bs)
    qs = string_to_array(cfgs.qs, dtype='float')
    Is = string_to_array(cfgs.Is, dtype='float')
    recons = reconstruct_quantize_custom(lX, lY, min_val, max_val, bs, qs, cfgs.Delta, Is)

    print('Reconstructed value set:')
    print(np.unique(recons))
    
    # evaluate
    R = np.array(R.todense())
    tr_mask = np.array(tr_mask.todense()).astype(np.float32)
    te_mask = np.array(te_mask.todense()).astype(np.float32)
    
    rmse_tr = RMSE(recons, R, tr_mask)
    rmse_te = RMSE(recons, R, te_mask)
    mae_tr = MAE(recons, R, tr_mask)
    mae_te= MAE(recons, R, te_mask)

    print('--------------------------------')
    print('RMSE (train - test): %f - %f' %(rmse_tr, rmse_te))
    print('MAE  (train - test): %f - %f' %(mae_tr, mae_te))
    print('--------------------------------')
    
if __name__ == '__main__':
    tf.app.run()
