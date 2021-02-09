

# These layers are commonly referred to as “dense” layers in papers,
# literature, and code, but you will occasionally see them called 
# fully-connected or “fc” for short in code or reference matetial.

# Our dense layer class will begin with two methods.



import numpy as np

class Layer_Dense:
    
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # initialize the weights and biases
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        # add the random initialization of weights and biases:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(1, n_neurons)


    # forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases

        self.output = np.dot(inputs, self.weights) + self.biases
