
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

pickle_file = 'notMNIST.pickle'
