import tensorflow as tf

def train_softmax(NUM_CATEGORIES, X_to_train, y_to_train, X_to_test, y_to_test, iterations, learning_rate=0.002):
    """train the model"""

    N = X_to_train.shape[1]

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
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # start the session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train with 1000 iterations
    for i in range(iterations):
        _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x_train: X_to_train, y_train: y_to_train})
        if i % 100 == 0:
            train_acc_val = sess.run(accuracy, feed_dict={x_train: X_to_train, y_train: y_to_train})
            test_acc_val = sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test})

            print('loss = ' + str(loss_val))
            print('train acc = ' + str(train_acc_val))
            print('test acc = ' + str(test_acc_val))

    return sess, y_predict, x_train, y_train

def train_nn(NUM_CATEGORIES, X_to_train, y_to_train, X_to_test, y_to_test, layers, iterations, learning_rate=0.0005):
    """train the model"""

    N = X_to_train.shape[1]

    # the input vector
    x_train = tf.placeholder(tf.float32, [None, N])

    # the ground truth vector
    y_train = tf.placeholder(tf.float32, [None, NUM_CATEGORIES])

    layer_i = x_train
    for i in layers:
        layer_i = tf.layers.dense(layer_i, i)
        layer_i = tf.nn.relu(layer_i)

    y_predict = tf.layers.dense(layer_i, NUM_CATEGORIES)

    # the loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_train))

    # the classifier
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # start the session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train with 1000 iterations
    for i in range(iterations):
        _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x_train: X_to_train, y_train: y_to_train})
        if i % 100 == 0:
            train_acc_val = sess.run(accuracy, feed_dict={x_train: X_to_train, y_train: y_to_train})
            test_acc_val = sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test})

            print('loss = ' + str(loss_val))
            print('train acc = ' + str(train_acc_val))
            print('test acc = ' + str(test_acc_val))

    return sess, y_predict, x_train, y_train