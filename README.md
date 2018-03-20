# neural-network

An implementation of a feedforward neural network. [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) is used for linear algrebra calculations. Test are implemented using [Catch](https://github.com/catchorg/Catch2).

The network can now overfit training data correctly. However, the implementation is currently too slow to train networks on large training sets. The next goal is to add dynamic step size updating in gradient descent to improve training speed, and to optimize matrix computations where possible.

The example in /examples uses data from the MNIST data base of handwritten digits, which can be found [here](http://yann.lecun.com/exdb/mnist/).