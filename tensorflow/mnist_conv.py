"""MNIST with tensorflow"""

import tensorflow as tf

from .utils import download_data

def weight_variable(shape):
    """
    define weight variable with random value from Gaussian distribution
    with mean=0 and std=0.1
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """define bias and initialize all values to 0.1"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """define a convolutional layer"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """define a pool layer"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

def train(mnist):
    """train the model with MNIST"""

    x_train = tf.placeholder(tf.float32, shape=[None, 784])
    y_train = tf.placeholder(tf.float32, shape=[None, 10])

    # reshape input
    x_image = tf.reshape(x_train, [-1, 28, 28, 1])

    # first convolutional layer with 32 filters, each 5x5x1 size
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = conv2d(x_image, W_conv1) + b_conv1

    # add relu layer
    h_relu1 = tf.nn.relu(h_conv1)

    # add max pooling layer
    h_pool1 = max_pool_2x2(h_relu1)

    # second convolutional layer with 64 filters, each 5x5x32 size
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2

    # add relu layer
    h_relu2 = tf.nn.relu(h_conv2)

    # add max pooling layer
    h_pool2 = max_pool_2x2(h_relu2)

    # flatten the layer to make it compatible with fully connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # add fully connected layer with 1024 neurons, each with size 7x7x64
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

    # add relu layer
    h_relu3 = tf.nn.relu(h_fc1)

    # add softmax layer with size 1024x10
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_relu3, W_fc2) + b_fc2

    '''
    training with cross entropy loss function
    and adam solver (based on SGD)
    '''

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # for model evaluation
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize all variables
    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())

    iterations = 1000
    batch_size = 100

    for i in range(iterations):
        batch = mnist.train.next_batch(batch_size)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x_train: batch[0], y_train: batch[1]})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x_train: batch[0], y_train: batch[1]})

    print("calculating accuracy...")
    eval_accuracy = calculate_accuracy_batch(accuracy, mnist, x_train, y_train)
    print("test accuracy %g"%eval_accuracy)

def calculate_accuracy_batch(accuracy, mnist, x_train, y_train):
    """calculate accuracy by batch"""
    import math

    batch_size = 500
    example_size = get_example_size(mnist.test.images)
    iterations = math.ceil(example_size / batch_size)
    weighted_accuracy = 0

    for _ in range(iterations):
        batch = mnist.test.next_batch(batch_size)
        test_accuracy = accuracy.eval(feed_dict={
            x_train: batch[0], y_train: batch[1]})
        weighted_accuracy += get_example_size(batch[0]) * test_accuracy

    return weighted_accuracy / example_size

def get_example_size(tensor):
    """get number of examples, i.e the first dimension"""
    return tensor.shape[0]

def run():
    """run the application"""
    mnist = download_data()
    train(mnist)

if __name__ == '__main__':
    run()
