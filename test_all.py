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

model_names = ['deepsea', 'basset', 'deepsea_custom', 'basset_custom']
activations = ['relu', 'exponential']
trial = 0

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


with open(os.path.join(results_path, 'results.txt'), 'w') as fout:

    results = []
    for model_name in model_names:
        for activation in activations:
            tf.keras.backend.clear_session()
            # build model
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
            print(model_name + '_' + activation)

            # compile model model
            auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
            aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
            model.compile(tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=[auroc, aupr])

            # load model params
            name = model_name + '_' + activation + '_' + str(trial)
            model_dir = os.path.join(results_path, name+'_weights.h5')
            model.load_weights(model_dir)

            # get test performance
            test_results = model.evaluate(x_test, y_test, batch_size=100)
        
            # save results
            fout.write("%s\t%s\t%.4f\t%.4f\n"%(model_name, activation, test_results[1], test_results[2]))



