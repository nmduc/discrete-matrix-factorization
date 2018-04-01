import tensorflow as tf
import numpy as np 
from models.dmfd import DMFD
from utils.data_loader import DataLoader
import time 
import configs.configs_dmfd as configs

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./data/MovieLens1M/", "Data directory.")
tf.flags.DEFINE_string("output_basedir", "./outputs/", "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("pretrained_fname", "", "Name of the pretrained model checkpoints (to resume from)")
tf.flags.DEFINE_integer("n_epochs", 1000, "Number of training epochs.")
tf.flags.DEFINE_integer("log_every_n_steps", 10,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("save_every_n_epochs", 100,
                        "Frequency at which session is saved.")
tf.flags.DEFINE_boolean("log_time", False, "Whether to print out running time or not")

FLAGS.output_dir = FLAGS.output_basedir + 'snapshots/snapshot'
FLAGS.log_dir = FLAGS.output_basedir + 'log/'

cfgs = configs.CONFIGS

def get_lambda_(lambda_, anneal_rate, epoch_counter, schedule='step', step=30):
    if schedule == 'exp':
        lambda_ = lambda_ * anneal_rate
    elif schedule == 'step':
        if (epoch_counter > 0) and (epoch_counter % step == 0):
            lambda_ = lambda_ * 25
    return lambda_ 

def main(unused_argv):
    assert FLAGS.output_dir, "--output_dir is required"
    # Create training directory.
    output_dir = FLAGS.output_dir
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)

    dl = DataLoader(FLAGS.data_dir)
    dl.load_data()
    dl.split()

    x_dim = dl.get_X_dim()
    y_dim = dl.get_Y_dim()

    # Build the model.
    model = DMFD(x_dim, y_dim, dl.min_val, dl.max_val, cfgs, log_dir=FLAGS.log_dir)

    if FLAGS.pretrained_fname:
        try:
            model.restore(FLAGS.pretrained_fname)
            print('Resume from %s' %(FLAGS.pretrained_fname))
        except:
            pass
    
    lr = cfgs.initial_lr
    epoch_counter = 0
    ite = 0
    lambda_ = cfgs.base_lambda_
    while True:
        start = time.time()
        x, y, R, mask, flag = dl.next_batch(cfgs.batch_size_x, cfgs.batch_size_y, 'train')
        load_data_time = time.time() - start
        if flag: 
            epoch_counter += 1
        
        # some boolean variables    
        do_log = (ite % FLAGS.log_every_n_steps == 0)
        do_snapshot = flag and epoch_counter > 0 and epoch_counter % FLAGS.save_every_n_epochs == 0
        val_loss = -1

        # train one step
        get_summary = do_log and cfgs.write_summary
        start = time.time()
        loss, _, summary, ite = model.partial_fit(x, y, R, mask, lr, lambda_, get_summary)
        one_iter_time = time.time() - start
        
        # writing outs
        if do_log:
            print('Iteration %d, (lr=%f, lambda_=%f) training loss  : %f' %(ite, lr, lambda_, loss))
            if FLAGS.log_time:
                print('Iteration %d, data loading: %f(s) ; one iteration: %f(s)' %(ite, load_data_time, one_iter_time))
            if cfgs.write_summary:
                model.log(summary)

        if do_snapshot:
            print('Snapshotting')
            model.save(FLAGS.output_dir)
        
        if flag: 
            lambda_ = get_lambda_(lambda_, cfgs.anneal_rate, epoch_counter - 1, cfgs.sigmoid_schedule)
            print('Finished epoch %d' %epoch_counter)
            print('--------------------------------------')
            if epoch_counter == FLAGS.n_epochs:
                if not do_snapshot:
                    print('Final snapshotting')
                    model.save(FLAGS.output_dir)
                break
            if epoch_counter % cfgs.num_epochs_per_decay == 0:
                lr = lr * cfgs.lr_decay_factor
                print('Decay learning rate to %f' %lr)

if __name__ == '__main__':
    tf.app.run()


