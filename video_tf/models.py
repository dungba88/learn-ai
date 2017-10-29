import tensorflow as tf

def train_softmax(X_to_train, X_to_test, y_to_train, y_to_test):
    """train the model"""

    N = X.shape[1]

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

    return y_predict, y_train

def train_nn(X_to_train, X_to_test, y_to_train, y_to_test):
    """train the model"""

    N = X.shape[1]

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

    return y_predict, y_train
