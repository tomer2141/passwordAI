from nn import NeuralNetwork
import random
nn = NeuralNetwork(2, 2, 1)


class TrainingData:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets


data = []
data.append(TrainingData([0, 0], [0]))
data.append(TrainingData([1, 1], [0]))
data.append(TrainingData([0, 1], [1]))
data.append(TrainingData([1, 0], [1]))

# The XOR problem
times = 100000
nowPrecent = 0
for i in range(times):
    rnd = random.randint(0, len(data) - 1)
    nn.feed(data[rnd].inputs, True, data[rnd].targets)
    if i > 0:
        precent = int(i / 1000)
        if precent != nowPrecent:
            nowPrecent = precent
            print(str(precent) + "% Done")

print("Training Done")
for x in data:
    print("Inputs " + str(x.inputs))
    print("Target " + str(x.targets))
    answer = nn.feed(x.inputs, False)
    print("Answer " + str(answer))
