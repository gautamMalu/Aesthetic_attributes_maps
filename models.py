from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dropout, Lambda, GlobalAveragePooling2D, merge, Input, Dense
from keras.models import Model
import keras.backend as K
#from keras.utils.visualize_util import plot
#from SpatialPyramidPooling import SpatialPyramidPooling


def l2_normalize(x):
    return K.l2_normalize(x, 0)


def l2_normalize_output_shape(input_shape):
    return input_shape


def squared_root_normalization(x):
    """
    Squared root normalization for convolution layers` output
    first apply global average pooling followed by squared root on all elements
    then l2 normalize the vector

    :param x: input tensor, output of convolution layer
    :return:
    """
    x = GlobalAveragePooling2D()(x)
    #output shape = (None, nc)
   # x = K.sqrt(x)
    #x = K.l2_normalize(x, axis=0)
    return x


def squared_root_normalization_output_shape(input_shape):
    """
    Return the output shape for squared root normalization layer
    for any given input size of the convolution filter
    :param input_shape: shape of the input
    :return: output shape
    """
    return (input_shape[0], input_shape[-1])


def model1(weights_path=None):
    '''
    Basic ResNet-FT for baseline comparisions.
    Creates a model by for all aesthetic attributes along
    with overall aesthetic score, by finetuning resnet50
    :param weights_path: path of the weight file
    :return: Keras model instance
    '''
    _input = Input(shape=(299, 299, 3))
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=_input)

    last_layer_output = GlobalAveragePooling2D()(resnet.get_layer('activation_49').output)

    # output of model
    outputs = []
    attrs = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF',
             'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
    for attribute in attrs:
        outputs.append(Dense(1, init='glorot_uniform', activation='tanh', name=attribute)(last_layer_output))

    non_negative_attrs = ['Repetition', 'Symmetry', 'score']
    for attribute in non_negative_attrs:
        outputs.append(Dense(1, init='glorot_uniform', activation='sigmoid', name=attribute)(last_layer_output))

    model = Model(input=_input, output=outputs)
    if weights_path:
        model.load_weights(weights_path)
    return model


def model2(weights_path=None):
    '''
    Creates a model by concatenating the features from lower layers
    with high level convolution features for all aesthetic attributes along
    with overall aesthetic score
    :param weights_path: path of the weight file
    :return: Keras model instance
    This is the model used in the paper
    '''
    _input = Input(shape=(299, 299, 3))
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=_input)
    activation_layers = []
    layers = resnet.layers
    for layer in layers:
        #  print layer.name, layer.input_shape, layer.output_shape
        if 'activation' in layer.name:
            activation_layers.append(layer)

    activations = 0
    activation_plus_squared_outputs = []
    # Remove last activation layer so
    # it can be used with spatial pooling layer if required
    nlayers = len(activation_layers) - 1
    for i in range(1, nlayers):
        layer = activation_layers[i]
        if layer.output_shape[-1] > activation_layers[i - 1].output_shape[-1]:
            #         print layer.name, layer.input_shape, layer.output_shape
            activations += layer.output_shape[-1]
            _out = Lambda(squared_root_normalization,
                          output_shape=squared_root_normalization_output_shape, name=layer.name + '_normalized')(layer.output)
            activation_plus_squared_outputs.append(_out)

            #  print "sum of all activations should be {}".format(activations)

    last_layer_output = GlobalAveragePooling2D()(activation_layers[-1].output)

   # last_layer_output = Lambda(K.sqrt, output_shape=squared_root_normalization_output_shape)(last_layer_output)
    last_layer_output = Lambda(l2_normalize, output_shape=l2_normalize_output_shape,
                               name=activation_layers[-1].name+'_normalized')(last_layer_output)

    activation_plus_squared_outputs.append(last_layer_output)

    merged = merge(activation_plus_squared_outputs, mode='concat', concat_axis=1)
    merged = Lambda(l2_normalize, output_shape=l2_normalize_output_shape, name='merge')(merged)

    # output of model
    outputs = []
    attrs = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF',
             'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
    for attribute in attrs:

        outputs.append(Dense(1, init='glorot_uniform', activation='tanh', name=attribute)(merged))

    non_negative_attrs = ['Repetition', 'Symmetry', 'score']
    for attribute in non_negative_attrs:
        outputs.append(Dense(1, init='glorot_uniform', activation='sigmoid', name=attribute)(merged))

    model = Model(input=_input, output=outputs)
    if weights_path:
        model.load_weights(weights_path)
    return model


if __name__ == '__main__':
    model = model2()
    model.summary()
  #  plot(model, to_file='model2.png', show_shapes=True)
