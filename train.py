import h5py, os
import numpy as np
from six.moves import cPickle
import argparse
import tensorflow as tf
import model_zoo

#-----------------------------------------------------------------
def _parse_example(serialized: bytes):
    """Return tuple of (features, labels) for a single example 
    in a TFRecord file.
    
    This function should not be used directly. Rather, it should be
    used in a tf.data.Dataset transformation.
    """
    features = {
        "feature/value": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "label/value": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    }
    example = tf.io.parse_single_example(serialized, features)
    # The keys in the `example` dictionary are arbitrary and are 
    # chosen when creating the TFRecord file.
    x = tf.io.decode_raw(example["feature/value"], tf.float32)
    y = tf.io.decode_raw(example["label/value"], tf.float32)
    # The shapes are encoded in the TFRecord file, but we cannot use
    # them dynamically (aka reshape according to the shape in this example).
    x = tf.reshape(x, shape=[1000, 4])
    y = tf.reshape(y, shape=[919])
    return x, y

def get_validation_arrays(path):
    """Return numpy arrays of data.

    Parameters
    ----------
    path : str, Path-like
        Path to HDF5 file. Must have the following datasets:
            - /deepsea/validation/features
            - /deepsea/validation/features

    Returns
    -------
    Numpy arrays: (x_valid, y_valid)
    """
    with h5py.File(path, "r") as f:
        x_valid = f["/deepsea/validation/features"][:]
        y_valid = f["/deepsea/validation/labels"][:]

    x_valid = x_valid.astype(np.float32)
    y_valid = y_valid.astype(np.float32)

    return tf.data.Dataset.from_tensor_slices((x_valid, y_valid)) 

#-----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, default=0.05, help="model_name")
parser.add_argument("-a", type=str, default='relu', help="activation")
parser.add_argument("-t", type=int, default=None, help="trial")
args = parser.parse_args()

model_name = args.m
activation = args.a
trial = args.t

# set paths
results_path = '../results_deepsea'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# load data
data_path = '../../data'


tfrec_glob = os.path.join(data_path, 'tfrecord', 'deepsea_train_shard-*.tfrec')
batch_size = 64
num_parallel_calls = 4
dset = tf.data.Dataset.list_files(tfrec_glob, shuffle=True)
dset = dset.interleave(
    lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP"),
    num_parallel_calls=num_parallel_calls,
    cycle_length=num_parallel_calls,
)
dset = dset.map(_parse_example, num_parallel_calls=num_parallel_calls)
dset = dset.shuffle(10000, reshuffle_each_iteration=True)
dset = dset.batch(batch_size)
dset = dset.prefetch(batch_size)

filepath = os.path.join(data_path, 'deepsea_dataset.h5')
validset = get_validation_arrays(filepath)
valid_set = validset.batch(batch_size)

# get shapes
L = 1000
A = 4
num_labels = 919

# build model
print(model_name)
if model_name == 'deepsea':
    model = model_zoo.deepsea((L,A), num_labels, activation)
elif model_name == 'danq':
    model = model_zoo.danq((L,A), num_labels, activation)
elif model_name == 'basset':
    model = model_zoo.basset((L,A), num_labels, activation)
elif model_name == 'deepatt':
    model = model_zoo.deepatt((L,A), num_labels, activation)
elif model_name == 'cnn_att':
    model = model_zoo.cnn_att((L,A), num_labels, activation)
elif model_name == 'cnn_lstm_trans_1':
    model = model_zoo.cnn_lstm_trans((L,A), num_labels, activation, num_layers=1)
elif model_name == 'cnn_lstm_trans_2':
    model = model_zoo.cnn_lstm_trans((L,A), num_labels, activation, num_layers=2)
elif model_name == 'cnn_lstm_trans_4':
    model = model_zoo.cnn_lstm_trans((L,A), num_labels, activation, num_layers=4)
elif model_name == 'cnn_trans_1':
    model = model_zoo.cnn_trans((L,A), num_labels, activation, num_layers=1)
elif model_name == 'cnn_trans_2':
    model = model_zoo.cnn_trans((L,A), num_labels, activation, num_layers=2)
elif model_name == 'cnn_trans_4':
    model = model_zoo.cnn_trans((L,A), num_labels, activation, num_layers=4)
elif model_name == 'deepsea_custom':
    model = model_zoo.deepsea_custom((L,A), num_labels, activation)
elif model_name == 'danq_custom':
    model = model_zoo.danq_custom((L,A), num_labels, activation)
elif model_name == 'basset_custom':
    model = model_zoo.basset_custom((L,A), num_labels, activation)
else:
    print("can't find model")

model_name = model_name + '_' + activation + '_' + str(trial)

# compile model model
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
model.compile(tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=[auroc, aupr])

# early stopping callback
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auroc', 
                                                patience=6, 
                                                verbose=1, 
                                                mode='max', 
                                                restore_best_weights=True)
# reduce learning rate callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auroc', 
                                                factor=0.2,
                                                patience=3, 
                                                min_lr=1e-7,
                                                mode='max',
                                                verbose=1) 

# train model
history = model.fit(dset, 
                    epochs=100,
                    validation_data=valid_set, 
                    validation_steps=1000,
                    callbacks=[es_callback, reduce_lr])


# save model params
model_dir = os.path.join(results_path, model_name+'_weights.h5')
model.save_weights(model_dir)
