
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

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
num_channels = 1 # grayscale


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
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


batch_size = 32
patch_size = 5
depth = 16
num_hidden_1 = 256
num_hidden_2 = 128

graph = tf.Graph()

with graph.as_default():

  # Input data.
    tf_dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
    tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    drop_ratio = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    lrate = tf.placeholder(tf.float32)

    xavier = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    xavier_conv = tf.contrib.layers.xavier_initializer_conv2d()

    conv_1 = tf.layers.conv2d(tf_dataset, 16, 4, 2, padding='same', kernel_initializer=xavier_conv, activation=tf.nn.elu)
    pooling_1 = tf.layers.max_pooling2d(conv_1, 4, 2)
    conv_2 = tf.layers.conv2d(pooling_1, 32, 4, 1, padding='same', kernel_initializer=xavier_conv, activation=tf.nn.elu)
    pooling_2 = tf.layers.max_pooling2d(conv_2, 4, 1)
    conv_3 = tf.layers.conv2d(pooling_2, 64, 2, 1, padding='same', kernel_initializer=xavier_conv, activation=tf.nn.elu)
    pooling_3 = tf.layers.max_pooling2d(conv_3, 2, 1)
    shape = tf_dataset.get_shape().as_list()
    input_line = tf.reshape(tf_dataset, [-1, shape[1]*shape[2]*shape[3]])
    rough_1 = tf.layers.dense(input_line, 1024, activation=tf.nn.elu,  kernel_initializer=xavier)
    rdropout_1 = tf.layers.dropout(rough_1, rate=drop_ratio, training=is_training)
    rough_2 = tf.layers.dense(rdropout_1, 512, activation=tf.nn.elu,  kernel_initializer=xavier)
    rdropout_2 = tf.layers.dropout(rough_2, rate=drop_ratio, training=is_training)
    rough_3 = tf.layers.dense(rdropout_2, 128, activation=tf.nn.elu,  kernel_initializer=xavier)
    rdropout_3 = tf.layers.dropout(rough_3, rate=drop_ratio, training=is_training)

    last_pooling = pooling_3
    shape = last_pooling.get_shape().as_list()
    last_pooling = tf.reshape(last_pooling, [-1, shape[1]*shape[2]*shape[3]])
    dense_1 = tf.layers.dense(last_pooling, num_hidden_1, activation=tf.nn.elu,  kernel_initializer=xavier)
    dropout_1 = tf.layers.dropout(dense_1, rate=drop_ratio, training=is_training)
    dense_2 = tf.layers.dense(dropout_1, num_hidden_2, activation=tf.nn.elu,  kernel_initializer=xavier)
    dropout_2 = tf.layers.dropout(dense_2, rate=drop_ratio, training=is_training)

#    concat = tf.concat([sdropout_2, rdropout_3, dropout_2], 1)
    concat = tf.concat([rdropout_3, dropout_2], 1)
    dense_final = tf.layers.dense(concat,128, activation=tf.nn.elu,  kernel_initializer=xavier)
    dropout_final = tf.layers.dropout(dense_final, rate=drop_ratio, training=is_training)

    logits = tf.layers.dense(dropout_final, num_labels, activation=None,  kernel_initializer=xavier)






    loss = tf.losses.softmax_cross_entropy(tf_labels, logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)
    prediction = tf.nn.softmax( logits )



num_steps = 20001
learning_schedule = {0: 0.001, 10000: 0.0005, 15000: 0.0001}

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Problem 3 Initialized")
    for step in range(num_steps):
        for k in learning_schedule.keys():
            if step >= k:
                lr = learning_schedule[k]
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels, drop_ratio: 0.5, is_training: True, lrate:lr}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)


        if (step % 500 == 0):
            print('step {}'.format(step))
#            feed_dict = {tf_dataset : train_dataset, drop_ratio: 1, is_training: False}
#            predictions = session.run([prediction], feed_dict=feed_dict)
#            print('Training accuracy: {0:.1f}%'.format(accuracy(np.array(predictions[0]), train_labels)))

            feed_dict = {tf_dataset : valid_dataset, drop_ratio: 1, is_training: False}
            predictions = session.run([prediction], feed_dict=feed_dict)
            print('Validation accuracy: {0:.1f}%'.format(accuracy(np.array(predictions[0]), valid_labels)))

            feed_dict = {tf_dataset : test_dataset, drop_ratio: 1, is_training: False}
            predictions = session.run([prediction], feed_dict=feed_dict)
            print('Test accuracy: {0:.01f}%'.format( accuracy(np.array(predictions[0]), test_labels)) )


