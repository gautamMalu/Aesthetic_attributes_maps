import joblib
from sys import argv
from os.path import exists
from scipy.stats import spearmanr as spr
from sklearn.metrics import mean_squared_error as mse
from models import model2
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
import pandas as pd

def get_session(gpu_fraction=0.6):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if __name__ == '__main__': 
#    KTF.set_session(get_session())
    # mode = on which data we are testing     
    _, mode, weights_file, batch_size = argv
    batch_size = int(batch_size)
    mode = str(mode)
    assert (exists(weights_file))
    if mode == 'train':
        _file = '../data/testing/train.pkl'
        gt_file = 'imgListTrainRegression_.csv'
    elif mode == 'val':
        _file = '../data/testing/val.pkl'
        gt_file = 'imgListValidationRegression_.csv'
    else:
        _file = '../data/testing/test.pkl'
        gt_file = 'imgListTestNewRegression_.csv'

    assert(exists(_file))
    data = joblib.load(_file)
    groundTruth = pd.read_csv(gt_file, header=0, delimiter=',')
    n = groundTruth.shape[0]
    predAtt = pd.DataFrame(index=groundTruth.index, columns=groundTruth.columns)
    x = data[0]
    y_true = data[1]

    model = model2(weights_path=weights_file)
    y_predict = model.predict(x, batch_size=batch_size, verbose=1)

    attrs = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF',
             'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor', 'Repetition', 'Symmetry', 'score']
    for i,attr in enumerate(attrs):
	attr_true = y_true[attr]
	attr_predict = y_predict[i]
	rho, p_value = spr(attr_true, attr_predict)
        error = mse(attr_true, attr_predict)
        print "for {} the spr correlation: {} with p value {} and error value: {}".format(attr, rho, p_value, error)

        attr_predict = pd.Series(y_predict[i].reshape(n))
        predAtt[attr] = attr_predict.values
  
    predAtt['ImageFile'] = groundTruth['ImageFile']
    predAtt.to_csv(gt_file[0:-4]+'_predict.csv', index=False)


