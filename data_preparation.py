import numpy as np
from math import ceil
import pandas as pd
from sys import argv
import os
import joblib
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import h5py

def prepare_image(_image_src, target_size):
    '''
        Takes image source as input as return
        processed image array ready for train/test/val
    :param _image_src: source of image
    :return: image_array
    '''
    img = image.load_img(_image_src, size = target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

if __name__ == '__main__':
    # file = .csv file with image and attributes
    # mode = one of (train, test, val)
    # savemod = which format to save into pkl to h5
    # usage: python data_preparation.py imgListValidationRegression_.csv val pkl

    _, _file, mode, savemode = argv
    savemode = str(savemode)
    target_size = (299, 299)
    assert (os.path.exists(_file))
    assert(savemode in ['h5', 'pkl'])

    outdir = './data'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    imgSrc = './datasetImages' # path to directory of images

    inData = pd.read_csv(_file, header=0, delimiter=',')
    attributes = inData.columns.tolist()[1:]
    print "list of attributes: {}".format(attributes)
    # Reset the index in case its not proper
    inData.reset_index(drop=True, inplace=True)

    n = inData.shape[0]
    inputImages = inData['ImageFile'].tolist()
    imageData = np.zeros((n, target_size[0], target_size[1], 3)) # image data array
    inputImages = map(lambda f: os.path.join(imgSrc, f), inputImages)
    for i, _image in enumerate(inputImages):
        x = prepare_image(_image, target_size)
        imageData[i, :, :, :] = x
        if (i + 1) % 100 == 0:
            print i, _image, x.shape

    print 'imageData.shape: {}'.format(imageData.shape)
    n = imageData.shape[0]

    shuffled_index = np.arange(n)
    np.random.shuffle(shuffled_index)
    print 'shuffling data'

    imageData[:, :, :, :] = imageData[shuffled_index, :, :, :]
    # shuffled index and then reset the index
    inData = inData.iloc[shuffled_index]
    inData.reset_index(drop=True, inplace=True)

    outputs = {} # attributes groud truth array

    print "after shuffling: ", inData.columns
    for attribute in attributes:
        outputs[attribute] = inData[attribute].values

    data = [imageData, outputs]
    if savemode == 'h5':
        _filesave = str(mode) + '.h5'
        _filesave = os.path.join(outdir, _filesave)
        print 'saving data in {}'.format(_filesave)
        h5f = h5py.File(_filesave, 'w')
        h5f.create_dataset('x', data=imageData, chunks=True)
        h5f.create_dataset('y', data=outputs, chunks=True)
        h5f.close()
    else:
        _filesave = str(mode) + '.pkl'
        _filesave = os.path.join(outdir, _filesave)
        print "saving data in {}".format(_filesave)
        joblib.dump(data, _filesave, compress=True)
