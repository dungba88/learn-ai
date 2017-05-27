"""MNIST with tensorflow"""

def download_data():
    """download MNIST data"""
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

def train(mnist):
    """train the model with MNIST"""
    import tensorflow as tf
    # the input vector
    x_train = tf.placeholder(tf.float32, [None, 784])

    # weight and bias
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # the predicted output
    y_predict = tf.matmul(x_train, W) + b

    # the ground truth vector
    y_train = tf.placeholder(tf.float32, [None, 10])

    # the loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_train))

    # the classifier
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # start the session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x_train: batch_xs, y_train: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x_train: mnist.test.images, y_train: mnist.test.labels}))

def run():
    """run the application"""
    mnist = download_data()
    train(mnist)

if __name__ == '__main__':
    run()
