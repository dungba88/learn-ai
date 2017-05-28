"""utility module"""

def download_data():
    """download MNIST data"""
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("MNIST_data/", one_hot=True)
