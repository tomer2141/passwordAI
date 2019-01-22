from nn import NeuralNetwork
import random
nn = NeuralNetwork(2, 2, 1)


class TrainingData:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets


data = []
data.append(TrainingData([0, 0], [0]))
data.append(TrainingData([0, 1], [1]))
data.append(TrainingData([1, 0], [1]))
data.append(TrainingData([1, 1], [0]))

for i in range(100000):
    rnd = random.randint(0, len(data) - 1)
    nn.feed(data[rnd].inputs, True, data[rnd].targets)

print("Training Done")
while True:
    inputs = []
    first = float(input("First Element: "))
    inputs.append(first)
    second = float(input("second Element: "))
    inputs.append(second)
    print(inputs)
    answer = nn.feed(inputs, False)
    print(answer)
    print("------------------------------")
