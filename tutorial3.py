
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

"""

# Problem 1

beta = 0.01
batch_size = 128

graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(None, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
    weights = tf.Variable( tf.truncated_normal([image_size * image_size, num_labels]) )
    biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    regularization = tf.nn.l2_loss(weights)
    loss += regularization * beta

  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
    train_prediction = tf.nn.softmax(tf.matmul(tf_train_dataset, weights) + biases)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


num_steps = 3001

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
    tf.global_variables_initializer().run()
    print('Problem 1 Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.

        feed_dict = {tf_train_dataset :batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            feed_dict = {tf_train_dataset : train_dataset, tf_train_labels : train_labels}
            l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)

            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# Problem 2

beta = 0.01
batch_size = 128
train_dataset_small = train_dataset[:256]
train_labels_small = train_labels[:256]
graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(None, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
    weights = tf.Variable( tf.truncated_normal([image_size * image_size, num_labels]) )
    biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

  #  regularization = tf.nn.l2_loss(weights)
  #  loss += regularization * beta

  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
    train_prediction = tf.nn.softmax(tf.matmul(tf_train_dataset, weights) + biases)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


num_steps = 3001

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
    tf.global_variables_initializer().run()
    print("\n")
    print('Problem 2 Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels_small.shape[0] - batch_size)
    # Generate a minibatch.
        batch_data = train_dataset_small[offset:(offset + batch_size), :]
        batch_labels = train_labels_small[offset:(offset + batch_size), :]
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.

        feed_dict = {tf_train_dataset :batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            feed_dict = {tf_train_dataset : train_dataset_small, tf_train_labels : train_labels_small}
            l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)

            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels_small))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


#Problem 3
#beta = 0.1
n_hidden_nodes = 1024
train_dataset_small = train_dataset[:1024]
train_labels_small = train_labels[:1024]

graph = tf.Graph()
with graph.as_default():
    tf_dataset = tf.placeholder(tf.float32, shape=(None, image_size*image_size))
    tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    hidden_weights = tf.Variable( tf.truncated_normal([image_size*image_size, n_hidden_nodes]) )
    hidden_biases = tf.Variable( tf.zeros([n_hidden_nodes]) )
    #hidden_layer = tf.nn.relu( tf.matmul(tf_dataset, hidden_weights) + hidden_biases )
    hidden_layer = tf.nn.dropout( tf.nn.relu( tf.matmul(tf_dataset, hidden_weights) + hidden_biases ) , 0.5)

    output_weights = tf.Variable( tf.truncated_normal([n_hidden_nodes, num_labels]) )
    output_biases = tf.Variable( tf.zeros([num_labels]) )
    logits = tf.matmul(  hidden_layer, output_weights ) + output_biases
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=logits) )

  #  regularization = tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(hidden_weights)
  #  loss += beta * regularization
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    prediction = tf.nn.softmax( logits )


num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Problem 3 Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels_small.shape[0] - batch_size)
        batch_data = train_dataset_small[offset:(offset + batch_size), :]
        batch_labels = train_labels_small[offset:(offset + batch_size), :]
        feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)


        if (step % 500 == 0):
            feed_dict = {tf_dataset : train_dataset_small}
            predictions = session.run([prediction], feed_dict=feed_dict)

            print('Training accuracy: {0:.1f}%'.format(accuracy(np.array(predictions[0]), train_labels_small)))

            feed_dict = {tf_dataset : valid_dataset}
            predictions = session.run([prediction], feed_dict=feed_dict)
            print('Validation accuracy: {0:.1f}%'.format(accuracy(np.array(predictions[0]), valid_labels)))

            feed_dict = {tf_dataset : test_dataset}
            predictions = session.run([prediction], feed_dict=feed_dict)
            print('Test accuracy: {0:.01f}%'.format( accuracy(np.array(predictions[0]), test_labels)) )
"""
#Problem 4
"""

    # sklearn random forest model
random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(train_dataset, train_labels)
predicts = random_forest.predict(test_dataset)
score = metrics.accuracy_score(test_labels, predicts)
print("sklearn Random Forest accuracy score: {}".format(round(score * 100, 2)))

"""
beta = 0.1
n_hidden_nodes_1 = 2048
n_hidden_nodes_2 = 1024
n_hidden_nodes_2 = 512
batch_size = 128
graph = tf.Graph()
with graph.as_default():
    tf_dataset = tf.placeholder(tf.float32, shape=(None, image_size*image_size))
    tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    drop_ratio = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    xavier = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    hidden_layer_1_dense = tf.layers.dense(tf_dataset, n_hidden_nodes_1, activation=tf.nn.elu,  kernel_initializer=xavier)

    hidden_layer_1 = tf.layers.dropout(hidden_layer_1_dense, rate=drop_ratio, training=is_training)
    hidden_layer_2_dense = tf.layers.dense(hidden_layer_1, n_hidden_nodes_2, activation=tf.nn.elu,  kernel_initializer=xavier)

    hidden_layer_2 = tf.layers.dropout(hidden_layer_2_dense, rate=drop_ratio, training=is_training)

    hidden_layer_3_dense = tf.layers.dense(hidden_layer_2, n_hidden_nodes_3, activation=tf.nn.elu,  kernel_initializer=xavier)

    hidden_layer_3 = tf.layers.dropout(hidden_layer_3_dense, rate=drop_ratio, training=is_training)

    logits = tf.layers.dense(hidden_layer_3, num_labels, activation=None,  kernel_initializer=xavier)

    loss = tf.losses.softmax_cross_entropy(tf_labels, logits)

  #  regularization = tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(hidden_weights)
  #  loss += beta * regularization
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    prediction = tf.nn.softmax( logits )


num_steps = 6001
