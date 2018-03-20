# neural-network

An implementation of a feedforward neural network. [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) is used for linear algrebra calculations. Test are implemented using [Catch](https://github.com/catchorg/Catch2).

The network can currently overfit training data correctly. However, at this point the implementation is too slow to train networks on large training sets. The next goal is to add dynamic step size updating in gradient descent to improve training speed, and to optimize matrix computations where possible.