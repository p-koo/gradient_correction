import h5py, os
import numpy as np
from six.moves import cPickle
import argparse
import tensorflow as tf
import model_zoo

#-----------------------------------------------------------------


def get_data(path, dataset='test'):
    with h5py.File(path, "r") as f:
        x = f["/deepsea/"+dataset+"/features"][:].astype(np.float32)
        y = f["/deepsea/"+dataset+"/labels"][:].astype(np.float32)
    return x, y

#-----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, default='deepsea', help="model_name")
parser.add_argument("-a", type=str, default='relu', help="activation")
parser.add_argument("-t", type=int, default=0, help="trial")
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

filepath = os.path.join(data_path, 'deepsea_dataset.h5')
x_test, y_test = get_data(filepath, dataset='test')

# get shapes
N, L, A = x_test.shape
num_labels = y_test.shape[1]

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

# save model params
model_dir = os.path.join(results_path, model_name+'_weights.h5')
model.load_weights(model_dir)

model.evaluate(x_test, y_test)


