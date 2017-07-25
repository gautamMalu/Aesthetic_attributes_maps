from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import pandas as pd
from sys import argv
import cv2
from keras.layers.pooling import AveragePooling2D
from models import model1, model2, model3
import os
import joblib


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_image(path):
    img_path = path
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def deprocess(path):
    img_path = path
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    return x


def grad_cam(input_model, image, weights, feature_maps=None):
    #activation size of final convolition layer is 10x10"
    cam = np.ones((10, 10), dtype=np.float32)
    # Add weighted activation maps
    grads_val = weights
    for i in range(grads_val.shape[0]):
        # Added relu
        temp = (weights[i, :] * feature_maps[:, :, i])
        np.maximum(temp, 0, temp)
        cam += temp

    # resize and normalization
    del feature_maps
    cam = cv2.resize(cam, (299, 299))
    # Relu
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
   # print image.shape


    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)   
    cam = 0.5*np.float32(cam) + 0.5*np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)



def get_heatmaps(image_path, attrs):
    filename = image_path.split('/')[-1]
    preprocessed_input = load_image(image_path)
    for attr in attrs:
        cam = grad_cam(model, image=preprocessed_input, attr=attr)
        save_name = os.path.join(attr, filename)
        cv2.imwrite(save_name, cam)
        del cam


def get_features(image, model):
    '''
    get the feature map of all activation layer for given
    image and given model
    :param image: input image path
    :param model: given model
    :return: all activation layers features
    '''

   # image = load_image(image_src)
    feature_maps = np.zeros((10, 10, 15104))
    activation_layers = ['activation_' + str(i) for i in range(4, 50, 3)]
    start_index = 0

    for i, layer_name in enumerate(activation_layers):
        layer = model.get_layer(layer_name)
        nchannel = layer.output_shape[-1]
        conv_output = layer.output
	# Adujusting pooling size with respect to input layers` size
        if layer.output_shape[-2] == 74:
            conv_output = AveragePooling2D(pool_size=(7, 7))(conv_output)
        if layer.output_shape[-2] == 37:
            conv_output = AveragePooling2D(pool_size=(4, 4), border_mode='same')(conv_output)
        if layer.output_shape[-2] == 19:
            conv_output = AveragePooling2D(pool_size=(2, 2), border_mode='same')(conv_output)

        featuremap_function = K.function([model.input, K.learning_phase()], [conv_output])

        output = featuremap_function([image, 0])
        feature_maps[:, :, start_index:start_index+nchannel] = output[0][0, :, :, :]

        start_index = start_index + nchannel

    return feature_maps


if __name__ == '__main__':
    #usage: python visualization imgListTestNewRegression_.csv
    _, _file = argv
 
    #features_dir where to save features file
    # We save all the features in one go we don't 
    # have to generate features when genereting heatmaps
    # for different attributes
	
    features_dir = './feature_maps/' 
    imgSrc = 'datasetImages' # path to directory where all images are
    inData = pd.read_csv(_file, delimiter=',', header=0)
   
    inputImages = inData.ImageFile.tolist()
    inputImages = map(lambda f: os.path.join(imgSrc, f), inputImages)

   
    weight_path = 'weights-improvement__016-0.022715.hdf5'
    model = model2(weights_path=weight_path)

    attrs = ['ColorHarmony', 'Content', 'DoF',
              'Light', 'Object', 'VividColor', 'score']

    # make one directory for each attributes where we will save
    # the heatmap images 
    for attr in attrs:
        if not os.path.isdir(attr):
            os.makedirs(attr)

    n = len(inputImages)
    # weights contains weights for all attributes 
    # from the trained model, please see get_weights.py
    # to see how we have extracted weights from trained model.
    weights = joblib.load('weights.pkl')

    for i, image_path in enumerate(inputImages):
        img = load_image(image_path)

        filename = image_path.split('/')[-1]
        print "{}/{} {}".format(i, n_images, image_path)

        save_features = filename[:-3]+'pkl'
        save_features = os.path.join(features_dir, save_features)
        if os.path.exists(save_features):
            print "features exists no need to get features"
            features = joblib.load(save_features)
        else:
            print "Getting features"
            features = get_features(img, model)
            joblib.dump(features, save_features, compress=True)
	
       for attr in attrs:
            attr_val = inData.ix[i, attr]
            print attr, attr_val
            cam = grad_cam(model, img, weights[attr], attr_value=attr_val, feature_maps=features)
            save_name = os.path.join(attr,filename) # save path of heatmap image
            cv2.imwrite(save_name, cam)
        print '\n'
