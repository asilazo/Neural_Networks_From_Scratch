
# Activation functions are applied to the output of a 
# neuron (or layer of neurons), and they modify outputs.
# We use activation functions because if the activation function 
# itself is nonlinear, it allows for neural networks with usually
# two or more hidden layers to map nonlinear functions.

# In general, a neural network will have two types of activation 
# functions. The first will be the activation function used in 
# hidden layers, and the second will be used in the output layer.
# Usually, the activation function used for hidden neurons will
# be the same for all of them, but it doesn’t have to.


# ReLU activation code implemetations

import numpy as np

class Activation_ReLU:

    # forward pass
    def forward(self, inputs):
        ​# Calculate output values from input
        self.output = np.maximum(0, inputs)