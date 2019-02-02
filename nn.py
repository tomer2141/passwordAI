import numpy
import math


def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(x): return x * (1.0 - x)


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1. * (x > 0)


def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x))


# TODO: ADD SOFTMAX
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

    def feed(self, inputs, isTrain, targets=[]):
        # Determine which activision function to use
        sigmoidfunc = numpy.vectorize(sigmoid)
        dsigmoidfunc = numpy.vectorize(dsigmoid)
        relufunc = numpy.vectorize(ReLU)
        drelufunc = numpy.vectorize(dReLU)
        softmaxfunc = numpy.vectorize(softmax)

        # Generating the hidden nodes values
        inputs = numpy.array(inputs, dtype=float)
        inputs = inputs[:, numpy.newaxis]
        hidden = self.weigths_ih @ inputs
        hidden = hidden + self.bias_h
        # Activision function
        hidden = relufunc(hidden)  # vecfunc(hidden)

        # Generating the output nodes values
        outputs = self.weigths_ho @ hidden
        outputs = outputs + self.bias_o
        # Activision function
        outputs = sigmoidfunc(outputs)

        if isTrain:
            # Training
            if len(targets) == 0:
                return "Must provide targets while training"

            targets = numpy.array(targets)
            targets = targets[:, numpy.newaxis]
            # Calculate the outputs errors
            output_errors = targets - outputs

            # def tempsoftmax(z): return numpy.exp(z) / numpy.sum(numpy.exp(z))
            # output_errors = tempsoftmax(output_errors)

            # Calculate the gradient for hidden weights
            output_gradient = dsigmoidfunc(outputs)
            output_gradient = output_gradient * output_errors
            output_gradient = output_gradient * self.learningRate
            # Calculate the deltas
            weight_ho_deltas = output_gradient @ hidden.transpose()

            # Add the deltas to the ho weights
            self.weigths_ho = self.weigths_ho + weight_ho_deltas
            # Adjust the output bias
            self.bias_o = self.bias_o + output_gradient

            # ------------- NEXT LAYER --------------- #

            # Calculate the hidden errors
            hidden_errors = self.weigths_ho.transpose() @ output_errors
            # Calculate the gradient for inputs weights
            hidden_gradient = drelufunc(hidden)
            hidden_gradient = hidden_gradient * hidden_errors
            hidden_gradient = hidden_gradient * self.learningRate

            # Calculate the deltas
            inputs_mat = numpy.asmatrix(inputs)
            weight_iw_deltas = hidden_gradient @ inputs.transpose()

            # Add the deltas to the ho weights
            self.weigths_ih = self.weigths_ih + weight_iw_deltas
            # Adjust the hidden bias
            self.bias_h = self.bias_h + hidden_gradient
        else:
            return numpy.squeeze(numpy.asarray(outputs))
