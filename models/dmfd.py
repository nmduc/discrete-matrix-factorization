import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.losses as L
import tensorflow.contrib.keras as K
import numpy as np 

class DMFD():
    def __init__(self, x_dim, y_dim, min_val, max_val, cfg, phase='train', log_dir=None):
        ''' Initialize network
        Inputs:
            - x_dim: (int) dimension of rows (number of columns)
            - y_dim: (int) dimension of columns (number of rows)
            - min_val: minimum entry value
            - max_val: maximum entry value
            - cfg: configurations
            - phase: 'train' or 'test'
            - log_dir: directory to write log
        '''
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.min_val = min_val
        self.max_val = max_val
        self.parse_model_configs(cfg)
        self.lambda_ = tf.placeholder(tf.float32, [])

        # define placeholders
        if phase == 'train':
            self.x = tf.placeholder(tf.float32, [self.batch_size_x, self.x_dim])
            self.y = tf.placeholder(tf.float32, [self.batch_size_y, self.y_dim])
            self.R = tf.placeholder(tf.float32, [self.batch_size_x, self.batch_size_y])
            self.mask = tf.placeholder(tf.float32, [self.batch_size_x, self.batch_size_y])
        elif phase == 'test':
            # to feed data with random batch size
            self.x = tf.placeholder(tf.float32, [None, self.x_dim])
            self.y = tf.placeholder(tf.float32, [None, self.y_dim])
            self.R = tf.placeholder(tf.float32, [None, None])
            self.mask = tf.placeholder(tf.float32, [None, None])

        self.lr = tf.placeholder(tf.float32, [])    # learning rate
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')
        
        self.bs = self.build_quantization_boundaries(self.init_bs)

        # initializer
        self.initializer = tf.contrib.layers.xavier_initializer 

        # encoder 
        with tf.variable_scope('user'):
            self.inp_x = self.x
            self.latent_x = self.build_embedder(self.inp_x, self.u_hidden_sizes, 'user')
        with tf.variable_scope('movie'):
            self.inp_y = self.y
            self.latent_y = self.build_embedder(self.inp_y, self.v_hidden_sizes, 'movie')

        self.recons = self.build_recon_approx_sigmoid(self.latent_x, self.latent_y, 
            self.lambda_, self.bs, self.init_qs, self.Is, self.Delta)

        # loss functions
        self.recon_loss = self.build_recon_loss(self.R, self.recons, self.mask)
        self.boundary_loss = self.build_boundary_loss(self.bs, self.init_bs)

        self.total_loss = self.build_total_loss(
            [self.recon_loss, self.boundary_loss],
            self.loss_weights 
        )

        # build train_opt
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_opt = self.build_optimizer(self.optimizer, self.lr, self.total_loss)
        init = tf.global_variables_initializer()
    
        # session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(init)
        self.prepare_logger(log_dir)
        return None

    def prepare_logger(self, log_dir):
        # summary writer
        self.saver = tf.train.Saver(max_to_keep=10)
        # logging
        if log_dir:
            self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            tf.summary.scalar("0_total_loss", self.total_loss)
            tf.summary.scalar("1_recon_loss", self.recon_loss)
            tf.summary.scalar("2_boundary_loss", self.boundary_loss)
            tvars = tf.trainable_variables()
            for var in tvars:
                tf.summary.histogram(var.op.name, var)
            grads_tvars = self.optimizer.compute_gradients(self.total_loss, var_list=tvars)
            for grad_var_pair in grads_tvars:
                tf.summary.histogram(grad_var_pair[0].op.name, grad_var_pair[0])
            self.merged_summaries = tf.summary.merge_all()

    def parse_model_configs(self, cfg):
        ''' get model and training configurations from a given cfg object
        '''
        self.batch_size_x = cfg.batch_size_x
        self.batch_size_y = cfg.batch_size_y
        self.weight_decay = cfg.weight_decay
        self.optimizer = cfg.optimizer
        self.loss_weights = self.string_to_array(cfg.loss_weights, dtype='float')
        self.u_hidden_sizes = self.string_to_array(cfg.u_hidden_sizes, dtype='int')
        self.v_hidden_sizes = self.string_to_array(cfg.v_hidden_sizes, dtype='int')
        self.keep_prob = None
        if cfg.dropout_keep_prob > 0 and cfg.dropout_keep_prob <= 1:
            self.keep_prob = cfg.dropout_keep_prob
        self.latent_dim = self.u_hidden_sizes[-1]
        self.use_bn = cfg.use_bn
        self.init_bs = self.string_to_array(cfg.bs, dtype='float')
        self.init_qs = self.string_to_array(cfg.qs, dtype='float')
        self.Is = self.string_to_array(cfg.Is, dtype='float')
        self.Delta = cfg.Delta
        if cfg.activation_fn == 'relu':
            self.transfer = tf.nn.relu
        elif cfg.activation_fn == 'tanh':
            self.transfer = tf.nn.tanh
        elif cfg.activation_fn == 'sigmoid':
            self.transfer = tf.nn.sigmoid
        else:
            assert False, 'Invalid activation function'

    def build_embedder(self, inp, hidden_sizes, scope):
        ''' build layers to embed rows or columns into the embedding space
        '''
        latent = inp
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected], 
                weights_initializer=self.initializer(),
                biases_initializer=tf.constant_initializer(0),
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(self.weight_decay)):              
                for i in xrange(len(hidden_sizes)):
                    latent = slim.fully_connected(latent, hidden_sizes[i], scope='fc%d'%i)
                    if self.use_bn: 
                        latent = self.bn_layer(latent, scope='bn%d'%i)
                    if i < len(hidden_sizes) - 1: # do not put non-linearity after the last layer 
                        latent = self.transfer(latent)
                        if self.keep_prob :
                            latent = slim.dropout(latent, self.keep_prob, 
                                is_training=self.is_training, scope='dropout%d'%i)
        return latent

    def build_quantization_boundaries(self, init_bs):
        ''' Declare quantization boundaries
        '''
        self.bs = tf.Variable(init_bs, name='boundaries', trainable=True)
        return self.bs

    def build_recon_approx_sigmoid(self, latent_x, latent_y, k, bs, qs, Is, Delta):
        ''' Reconstruct the discrete matrix with an approximation function, conprised of sigmoid components
        '''
        l2_norm_lx = tf.nn.l2_normalize(latent_x, dim=1)
        l2_norm_ly = tf.nn.l2_normalize(latent_y, dim=1)
        mul = tf.matmul(l2_norm_lx, l2_norm_ly, transpose_b=True) # N x M
        mid = (self.max_val + self.min_val) / 2
        mul = mul * (mid - self.min_val) + mid
        recons = tf.zeros_like(mul)
        for i in xrange(len(Is)):
            vals = Delta * tf.nn.sigmoid(k*(mul - (bs[i]))) + Is[i]
            mask = tf.multiply(tf.cast(tf.greater_equal(mul, qs[i]), tf.float32), tf.cast(tf.less(mul, qs[i+1]), tf.float32))
            recons = recons + tf.multiply(vals, mask)
        return recons

    def build_recon_loss(self, R, recon, mask):
        ''' Define the reconstruction loss
        ''' 
        diff = R - recon
        sq_diff = diff * diff 
        loss = tf.reduce_sum(mask * sq_diff) / tf.reduce_sum(mask)
        loss = tf.sqrt(loss)
        return loss

    def build_boundary_loss(self, boundaries, init_boundaries):
        ''' Define the boundary loss
        '''
        loss = tf.reduce_mean(tf.square(boundaries - init_boundaries))
        return loss

    def build_total_loss(self, losses, weights):
        ''' Define the final training loss as weighted sum of component losses
        '''
        assert len(losses) == len(weights), 'Number of losses and weights are not the same'
        for i in xrange(len(losses)):
            tf.contrib.losses.add_loss(tf.multiply(weights[i], losses[i]))
        total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True)
        return total_loss

    def build_optimizer(self, optimizer, lr, loss):
        ''' Declare optimizer
        '''
        if optimizer == "SGD":
            self.optimizer = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        elif optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(lr)
        elif optimizer == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(lr)
        elif optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(lr)
        assert self.optimizer, 'Invalid learning algorithm'
        train_opt = self.optimizer.minimize(loss, global_step=self.global_step)
        return train_opt


    def partial_fit(self, x, y, R, mask, lr, lambda_, get_summary=False):
        ''' Perform one training iteration
        '''
        summary = None
        step = self.sess.run(self.global_step)
        if get_summary:
            loss, train_opt, summary = self.sess.run([self.total_loss, self.train_opt, self.merged_summaries], 
                feed_dict={self.x:x, self.y:y, self.R:R, self.mask:mask, self.lr:lr, self.is_training:True, self.lambda_:lambda_})
        else:
            loss, train_opt, recons = self.sess.run([self.total_loss, self.train_opt, self.recons], 
                feed_dict={self.x:x, self.y:y, self.R:R, self.mask:mask, self.lr:lr, self.is_training:True, self.lambda_:lambda_})
        return loss, train_opt, summary, step

    def embed_x(self, x):
        ''' Transform given x to the embedding space
        '''
        latent_x = self.sess.run((self.latent_x), feed_dict={self.x:x, self.lr:0, self.is_training:False})
        return latent_x

    def embed_y(self, y):
        ''' Transform given y to the embedding space
        '''
        latent_y = self.sess.run((self.latent_y), feed_dict={self.y:y, self.lr:0, self.is_training:False})
        return latent_y

    def calc_loss(self, x, y, R, mask, lambda_):
        ''' Calculate loss
        '''
        loss = self.sess.run(self.total_loss, feed_dict={self.x:x, 
            self.y:y, self.R:R, self.mask:mask, self.lr:0, self.is_training:False, self.lambda_:lambda_})
        return loss

    def get_boundaries(self):
        ''' Get the learned boundary values 
        '''
        boundaries = self.sess.run(self.bs)
        return boundaries

    def bn_layer(self, inputs, scope):
        ''' Batch-norm layer
        '''
        bn = tf.contrib.layers.batch_norm(inputs, is_training=self.is_training, 
            center=True, fused=False, scale=True, updates_collections=None, decay=0.9, scope=scope)
        return bn

    def save(self, save_path):
        self.saver.save(self.sess, save_path, global_step=self.sess.run(self.global_step))

    def restore(self, save_path):
        self.saver.restore(self.sess, save_path)

    def log(self, summary):
        self.writer.add_summary(summary, global_step=self.sess.run(self.global_step))

    def string_to_array(self, str, dtype='int'):
        arr = str.strip().split(',')
        for i in xrange(len(arr)):
            if dtype == 'int':
                arr[i] = int(arr[i])
            elif dtype == 'float':
                arr[i] = float(arr[i])
        return arr