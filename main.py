from nn import NeuralNetwork
nn = NeuralNetwork(2, 2, 1)

inputs = [1, 0]

output = nn.feedForward(inputs)
print(output)
