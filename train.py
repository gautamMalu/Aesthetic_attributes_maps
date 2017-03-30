import joblib
import numpy as np
from sys import argv
from random import randrange, sample

from keras.optimizers import SGD,Adagrad
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from models import model2
from keras.layers import MaxPooling2D
from keras.preprocessing import image

import json

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def get_session(gpu_fraction=0.75):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if __name__ == '__main__':
 #   KTF.set_session(get_session())
    _, params_file = argv
    with open(params_file, 'r') as fp:
        params = json.load(fp)
    print params
    nb_epoch = params['nb_epoch']
    initial_epoch = params['initial_epoch']
    # first 0.9 then 0.5 
    base_lr = params['base_lr']
    print 'starting training for {} epoch with initial epoch number: {}'.format(nb_epoch, initial_epoch)
    batch_size = params['batch_size']
    assert (K.image_dim_ordering() == 'tf')
    weights_path = params['weights_path']
 
    model = model2(weights_path=weights_path)
    adagrad = Adagrad(lr=base_lr)
    loss_weights = params['loss_weights']
    loss = {}
    metrics = {}
    attrs = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF',
             'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor', 'Repetition', 'Symmetry']
   
    for attr in attrs:
   	loss[attr] = 'mean_squared_error'
        metrics[attr] = 'mean_squared_error'

    loss['score'] = 'mean_squared_error'
    metrics['score'] = 'mean_squared_error'
    
    model.compile(loss=loss, optimizer=adagrad, metrics=metrics,
                  loss_weights=loss_weights)
    print "model ready loading data now"
    
    # Validation Data
    val_data_src = '../data/val.pkl'
    train_data_src = '../data/train.pkl'
    val_data = joblib.load(val_data_src)
    train_data = joblib.load(train_data_src)
    print 'loaded data'
    
    # CallBacks
    log_dir = params['log_dir']
    if not isdir(log_dir):
        makedirs(log_dir)

    model_dir = params['model_dir']
    if not isdir(model_dir):
        makedirs(model_dir)

    print "log_dir: {} model_dir: {}".format(log_dir, model_dir)
    filepath = model_dir+"weights-improvement__{epoch:03d}-{val_score_loss:.6f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)

    datagen = image.ImageDataGenerator(
        featurewise_center=False,
        horizontal_flip=True,
        fill_mode='nearest')
    print "starting training now"
    model.fit_generator(datagen.flow(train_data[0], train_data[1],batch_size=batch_size),8448, nb_epoch, verbose=2,
                              callbacks=[checkpoint, tb],#,lr_scheduler],
                              validation_data=(val_data[0], val_data[1]), max_q_size=10, nb_worker=2,
                              pickle_safe=True, initial_epoch=initial_epoch)
