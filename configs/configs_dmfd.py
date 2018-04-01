import tensorflow as tf

class ModelConfig():
    u_hidden_sizes = [2048, 1024]
    v_hidden_sizes = [2048, 1024]
    loss_weights = [1.0, 0.1] # weights for: reconstruction loss, boundary loss
    dropout_keep_prob = 0.2 
    use_bn = True
    activation_fn = 'relu'
    base_lambda_ = 0.1
    anneal_rate = 1.015
    # discretization parameters
    bs = [1.5,2.5,3.5,4.5]      # b_v in eq.(5)
    qs = [1.0,2.0,3.0,4.0,5.0]  # q_v in eq.(5)
    Is = [1.0,2.0,3.0,4.0]      # I_v in eq.(5)
    Delta = 1.0     # \Delta in eq.(5))

class TrainConfig(object):
    """Sets the default training hyperparameters."""
    batch_size_x = 300
    batch_size_y = 200 
    optimizer = "adam" # "SGD", "momentum", "adam", "adadelta", "rmsprop"

    initial_lr = 1e-2 # initial learning rate
    lr_decay_factor = 0.65
    num_epochs_per_decay = 50
    weight_decay = 0.0

    sigmoid_schedule = 'exp' # 'step'
    write_summary = False


def arr_to_string(arr):
    for i in xrange(len(arr)):
        arr[i] = str(arr[i])
    return ','.join(arr)

# model configs
tf.flags.DEFINE_string('u_hidden_sizes', arr_to_string(ModelConfig.u_hidden_sizes),'')
tf.flags.DEFINE_string('v_hidden_sizes', arr_to_string(ModelConfig.v_hidden_sizes),'')
tf.flags.DEFINE_string('loss_weights', arr_to_string(ModelConfig.loss_weights),'')
tf.flags.DEFINE_float('dropout_keep_prob', ModelConfig.dropout_keep_prob,'')
tf.flags.DEFINE_boolean('use_bn', ModelConfig.use_bn,'')
tf.flags.DEFINE_string('activation_fn', ModelConfig.activation_fn,'')
tf.flags.DEFINE_float('base_lambda_', ModelConfig.base_lambda_,'')
tf.flags.DEFINE_float('anneal_rate', ModelConfig.anneal_rate,'')
tf.flags.DEFINE_string('bs', arr_to_string(ModelConfig.bs),'')
tf.flags.DEFINE_string('qs', arr_to_string(ModelConfig.qs),'')
tf.flags.DEFINE_string('Is', arr_to_string(ModelConfig.Is),'')
tf.flags.DEFINE_float('Delta', ModelConfig.Delta,'')

# training configs
tf.flags.DEFINE_integer('batch_size_x', TrainConfig.batch_size_x,'')
tf.flags.DEFINE_integer('batch_size_y', TrainConfig.batch_size_y,'')
tf.flags.DEFINE_string('optimizer', TrainConfig.optimizer,'')
tf.flags.DEFINE_float('initial_lr', TrainConfig.initial_lr,'')
tf.flags.DEFINE_float('lr_decay_factor', TrainConfig.lr_decay_factor,'')
tf.flags.DEFINE_integer('num_epochs_per_decay', TrainConfig.num_epochs_per_decay,'')
tf.flags.DEFINE_float('weight_decay', TrainConfig.weight_decay,'')
tf.flags.DEFINE_string('sigmoid_schedule', TrainConfig.sigmoid_schedule,'')
tf.flags.DEFINE_boolean('write_summary', TrainConfig.write_summary,'')

CONFIGS = tf.app.flags.FLAGS
