from os import path
from sys import argv
import joblib
from models import model2

if __name__ == '__main__':
    #_, weight_file = argv
    weight_file = 'weights-improvement__016-0.022715.hdf5'
    assert(path.exists(weight_file))
    model = model2(weights_path=weight_file)
    attrs = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF',
             'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor', 'Repetition', 'Symmetry', 'score']
    weights = {}
    for attr in attrs:
        print "processing for {}".format(attr)
        weights[attr] = model.get_layer(attr).get_weights()[0]
        print weights[attr].shape
    print "Saving in weights.pkl"
    joblib.dump(weights, 'weights.pkl')

