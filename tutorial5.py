

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
#%matplotlib inline
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE


url = 'http://mattmahoney.net/dc/'
