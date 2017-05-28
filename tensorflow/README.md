# MNIST with Tensorflow

1. Softmax regression:
    - Loss function: cross-entropy
    - Optimizer: Mini-batch gradient descent
    - Learning rate: 0.5
    - Iterations: 1000
    
 Result: ~92% accuracy
 
 2. CNN:
     - Loss function: cross-entropy
     - Optimizer: Adam
     - Learning rate: 0.0001
     - Iterations: 1000
     - Architecture: 
         + Conv(32 5x5x1 filters, 1 stride, padding SAME)
         + ReLU
         + Max pooling 2x2 (2 stride, padding SAME)
         + Conv(64 5x5x32 filters, 1 stride, padding SAME)
         + ReLU
         + Fully connected(1024 neurons)
         + Softmax(10 classes)
         
Result: ~97.5%
