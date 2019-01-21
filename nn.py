import numpy
import math


def sigmoid(x): return 1 / (1 + math.exp(-x))


# Make the inputs a matrix with the number of rows as the inputs length
def fromArray(arr):
    newMatrix = []
    for x in arr:
        row = [x]
        newMatrix.append(row)

    return newMatrix


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, outputs_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.outputs_nodes = outputs_nodes
        self.learningRate = 0.1

        # Creating the initial matrix for the wieghts between input nodes and hidden nodes
        self.weigths_ih = numpy.random.rand(
            self.hidden_nodes, self.input_nodes)
        # Creating the initial matrix for the wieghts between hidden nodes and output nodes
        self.weigths_ho = numpy.random.rand(
            self.outputs_nodes, self.hidden_nodes)
        # The bias weights
        self.bias_h = numpy.random.rand(hidden_nodes, 1)
        self.bias_o = numpy.random.rand(outputs_nodes, 1)

    def feedForward(self, inputs, train):
        # Determine which activision function to use
        vecfunc = numpy.vectorize(sigmoid)

        # Generating the hidden nodes values
        inputs = fromArray(inputs)
        hidden = self.weigths_ih @ inputs
        hidden = hidden + self.bias_h
        # Activision function
        hidden = vecfunc(hidden)

        # Generating the output nodes values
        output = self.weigths_ho @ hidden
        output = output + self.bias_o
        # Activision function
        output = vecfunc(output)

        if train:
            return output
        else:
            return numpy.squeeze(numpy.asarray(output))


    def train(self, input, target):
        output = self.feedForward(self, input, True)
