import os
import glob
import scipy.io as sio
import tensorflow as tf
import numpy as np

from sklearn import cross_validation as cv

NUM_CATEGORIES = 27

"""
27 categories:
- Softmax, Adam: 60%, 1%
- 1 layer NN, Adam, 2000 iterations: 84%, 4%
- 2 layer NN, Adam, 2000 iterations: 83%, 13%
"""

def load_data(path, dict_key, N):
    filenames = glob.glob(os.path.join(path, '*.mat'))

    X = []
    y = []

    M = 0

    for filename in filenames:
        clss = int(filename.split('_')[0][len(path) + 2:]) - 1
        if clss < NUM_CATEGORIES:
            x = sio.loadmat(filename)[dict_key][:,:,:41]
            X.append(x.flatten().reshape(N, 1))
            y.append(one_hot(clss))
            M += 1

    return np.array(X).T.reshape(M, N), np.array(y).reshape(M, NUM_CATEGORIES)

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i] = 1
    return b

def train_softmax(X, y):
    """train the model"""

    M = X.shape[0]
    N = X.shape[1]

    X_to_train, X_to_test, y_to_train, y_to_test = cv.train_test_split(X, y, test_size=0.2, random_state=1)

    # the input vector
    x_train = tf.placeholder(tf.float32, [None, N])

    # the ground truth vector
    y_train = tf.placeholder(tf.float32, [None, NUM_CATEGORIES])

    # weight and bias
    W = tf.Variable(tf.zeros([N, NUM_CATEGORIES]))
    b = tf.Variable(tf.zeros([NUM_CATEGORIES]))

    # the predicted output
    y_predict = tf.matmul(x_train, W) + b

    # the loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_train))

    # the classifier
    train_step = tf.train.AdamOptimizer(0.002).minimize(cross_entropy)

    # start the session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # train with 1000 iterations
    for i in range(2000):
        _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x_train: X_to_train, y_train: y_to_train})
        if i % 100 == 0:
            print('loss = ' + str(loss_val))

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x_train: X_to_train, y_train: y_to_train}))
    print(sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test}))

def train_nn(X, y):
    """train the model"""

    M = X.shape[0]
    N = X.shape[1]

    X_to_train, X_to_test, y_to_train, y_to_test = cv.train_test_split(X, y, test_size=0.2, random_state=1)

    # the input vector
    x_train = tf.placeholder(tf.float32, [None, N])

    # the ground truth vector
    y_train = tf.placeholder(tf.float32, [None, NUM_CATEGORIES])

    layer1 = tf.layers.dense(x_train, 512)
    layer1_relu = tf.nn.relu(layer1)

    layer2 = tf.layers.dense(layer1_relu, 256)
    layer2_relu = tf.nn.relu(layer2)

    layer3 = tf.layers.dense(layer2_relu, 64)
    layer3_relu = tf.nn.relu(layer3)

    y_predict = tf.layers.dense(layer3_relu, NUM_CATEGORIES)

    # the loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_train))

    # the classifier
    train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)

    # start the session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # train with 1000 iterations
    for i in range(2000):
        _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x_train: X_to_train, y_train: y_to_train})
        if i % 100 == 0:
            print('loss = ' + str(loss_val))

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x_train: X_to_train, y_train: y_to_train}))
    print(sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test}))

def main():
    X, y = load_data('skeleton', 'd_skel', 20*3*41)
    train_nn(X, y)

if __name__ == '__main__':
    main()
