import os
import glob
import scipy.io as sio
import tensorflow as tf
import numpy as np

from sklearn import cross_validation as cv

import models

NUM_CATEGORIES = 27

def load_data(path, dict_key, N):
    filenames = glob.glob(os.path.join(path, '*.mat'))

    X = []
    y = []

    M = 0

    for filename in filenames:
        clss = int(filename.split('_')[0][len(path) + 2:]) - 1
        if clss < NUM_CATEGORIES:
            x = sio.loadmat(filename)[dict_key][:,:,:41]
            X.append(x.reshape(N))
            y.append(one_hot(clss))
            M += 1

    return np.array(X), np.array(y)

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i] = 1
    return b

def main():
    X, y = load_data('skeleton', 'd_skel', 20*3*41)
    X_to_train, X_to_test, y_to_train, y_to_test = cv.train_test_split(X, y, test_size=0.2, random_state=80)

    sess, y_predict, x_train, y_train = models.train_nn(NUM_CATEGORIES,
                                                        X_to_train, y_to_train, X_to_test, y_to_test,
                                                        layers=(256, 256),
                                                        iterations=1000,
                                                        learning_rate=0.001)

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x_train: X_to_train, y_train: y_to_train}))
    print(sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test}))

if __name__ == '__main__':
    main()
