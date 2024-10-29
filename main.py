import numpy as np
import h5py
import json

import signal
import readchar

# Save Neural Network Weights and Bias to file
def save(filename):
    print("Saving")
    with open(filename + '.json', 'w') as f:
        for i in range(1, len(NeuralNetwork.layer)-1):
            layer = NeuralNetwork.layer[i]
            layerDict = {}
            for j in range(len(layer.node)):
                node = layer.node[j]
                nodeDict = {"weight": node.weight.tolist(), "bias": node.bias.tolist()}
                layerDict[j] = nodeDict
            f.write(json.dumps({"neuralNetwork" : {"layer" : {i : layerDict}, "layerSize" : {i: len(layer.node)}}}))
        f.write(json.dumps({"learningRate" : BackPropagation.learningRate, "batchSize" : BackPropagation.batchSize}))
        f.close()

def load(filename, aNeuralNetwork, aBackPropagation):
    print("Loading")
    # Load Neural Network Weights and Bias from file
    with open(filename + '.json', 'r') as f:
        data = json.load(f)
        # print(data)
        for i in range(1, len(aNeuralNetwork.layer)-1):
            layer = aNeuralNetwork.layer[i]
            layerDict = data["neuralNetwork"]["layer"][str(i)]
            print(layerDict)
            for j in range(len(layer.node)):
                node = layer.node[j]
                nodeDict = layerDict[str(j)]
                node.weight = np.array(nodeDict["weight"])
                node.bias = np.array(nodeDict["bias"])
        aBackPropagation.learningRate = data["learningRate"]
        aBackPropagation.batchSize = data["batchSize"]
        f.close()

# Save when I press ctrl + c
def handler(signum, frame):
    msg = "Ctrl-c was pressed. Do you really want to exit? y/n "
    print(msg, end="", flush=True)
    res = readchar.readchar()
    save("Network")
    if res == 'y':
        print("")
        exit(1)
    else:
        print("", end="\r", flush=True)
        print(" " * len(msg), end="", flush=True) # clear the printed line
        print("    ", end="\r", flush=True)
 
signal.signal(signal.SIGINT, handler)


class Node:
    def __init__(self, weightSize, biasSize, batchSize):
        self.weight = np.zeros(weightSize)
        self.bias = np.zeros(biasSize)
        self.outputBatch = np.zeros( batchSize)
    def __str__(self):
        return "weight: " + str(self.weight) + " bias: " + str(self.bias)
    def randomize(self):
        self.weight = np.subtract(np.multiply(np.random.rand(len(self.weight)),2), 1)
        self.bias = np.subtract(np.multiply(np.random.rand(len(self.bias)),2), 1)
    def getWeight(self):
        return self.weight
    def getBias(self):
        return self.bias
    def setWeight(self, weight):
        self.weight = weight
    def setBias(self, bias):
        self.bias = bias
    def calc(self, inputBatch, function):
        # print("Node: ", inputBatch)
        for i in range(len(inputBatch)):
            #print(len(self.weight))
            #print("Nodei: ", inputBatch[i])
            self.outputBatch[i] = function(np.add(np.multiply(self.weight, inputBatch[i]), self.bias))
        return self.outputBatch
    def resetOutputBatch(self):
        self.saveIndex = -1
        self.outputBatch = np.zeros(len(self.outputBatch))
    def getOutputBatch(self):
        return self.outputBatch
    
class InputLayer:
    def calc(self, inputBatch):
        self.output = inputBatch
        return self.output

class Layer:
    def __init__(self, numberOfNodes, weightSize, biasSize, batchSize, function):
        self.node = [Node(weightSize, biasSize, batchSize) for i in range(numberOfNodes)]
        self.function = function
        self.name = "Layer"
        self.batchSize = batchSize
    def __str__(self):
        return "node: " + str(self.node)
    def randomize(self):
        for i in range(len(self.node)):
            self.node[i].randomize()
    def calc(self, inputBatch):
        self.output = np.zeros(shape=(len(inputBatch), len(self.node)))
        for i in range(len(self.node)):
            self.outputNode = self.node[i].calc(inputBatch, self.function)
            for j in range(self.batchSize):
                self.output[j][i] = self.outputNode[j]
        # print("Layer: ", inputBatch)
        # print("Layerso: ", self.output)
        return self.output
    
class NeuralNetwork:
    def __init__(self, inputLayer):
        self.layer = [inputLayer]
    def __str__(self):
        return "layer: " + str(self.layer)
    def addLayer(self, Layer):
        self.layer.append(Layer)
    def calc(self, inputBatch):
        self.output = inputBatch
        # print("NN: ", inputBatch)
        # print("NNso: ", self.output)
        for i in range(len(self.layer)):
            self.output = self.layer[i].calc(self.output)
        return self.output
    def randomize(self):
        for i in range(1, len(self.layer)):
            if self.layer[i].name == "Layer":
                self.layer[i].randomize()

class NormalizationLayer:
    def __init__(self):
        self.averageFactor = 0;
        self.name = "NormalizationLayer"
    def unDoLoss(self, loss):
        return np.multiply(loss, self.averageFactor)
    def decalc(self):
        return np.multiply(self.output, self.averageFactor)
    def function(self, input):
        # self.averageFactor = np.sum(input)
        # if self.averageFactor == 0:
        #     return input
        # assert self.averageFactor != 0, "NormalizationLayer: averageFactor is 0"
        return np.divide(input, np.sum(input))
    def calc(self, inputBatch):
        self.output = np.zeros(shape=(len(inputBatch), len(inputBatch[0])))
        for i in range(len(inputBatch)):
            self.averageFactor = np.sum(np.exp(inputBatch[i]))
            self.output[i] = np.divide(np.exp(inputBatch[i]), self.averageFactor)
            # self.averageFactor = np.sum(inputBatch[i])
            # if self.averageFactor == 0:
            #     return 0
            # assert self.averageFactor != 0, "NormalizationLayer: averageFactor is 0"
            # self.output[i] = np.divide(inputBatch[i], self.averageFactor)
        return self.output

class Functions:
    def ReLU(x):
        if isinstance(x, (list, np.ndarray)):
            return max(0, np.sum(x))
        else:
            return max(0, x)
    def sigmoid(x):
        if isinstance(x, (list, np.ndarray)):
            return 1 / (1 + np.exp(-np.sum(x)))
        else:
            return 1 / (1 + np.exp(-x))
    def tanh(x):
        if isinstance(x, (list, np.ndarray)):
            return np.tanh(np.sum(x))
        else:
            return np.tanh(x)

class BackPropagation:
    def __init__(self, neuralNetwork, learningRate, batchSize):
        self.neuralNetwork = neuralNetwork
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.inputBatch = np.zeros(batchSize)
        self.targetOutputBatch = np.zeros(batchSize)
    def __str__(self):
        return "neuralNetwork: " + str(self.neuralNetwork) + " learningRate: " + str(self.learningRate)
    def lossMeanSquaredError(self, y):
        return np.mean(np.square(np.subtract(y,self.yHat)))
    def lossSquaredError(self, y):
        return np.square(np.subtract(y,self.yHat))
    def lossCrossEntropy(self, y):
        return np.multiply(-self.yHat, np.log(y))
    def train(self, trainingData, trainingLabel, lossFunction):
        self.dLossdOut = np.zeros(self.batchSize)
        self.dOutdZ = np.zeros(self.batchSize)
        self.numberOfEndRescaleLayer = 0
        self.correctTotal = 0
        self.correct = 0
        self.accuracyHistory = []
        self.correctHistory = []
        for l in range(0, len(trainingData), self.batchSize):
            self.loss = 0
            self.correct = 0
            if l + self.batchSize > len(trainingData):
                self.inputBatch = trainingData[l:]
                self.targetOutputBatch = trainingLabel[l:]
            else:
                self.inputBatch = trainingData[l:l+self.batchSize]
                self.targetOutputBatch = trainingLabel[l:l+self.batchSize]
            self.neuralNetwork.calc(self.inputBatch)
            for i in range(len(self.neuralNetwork.layer)-1, 1,-1):
                if self.neuralNetwork.layer[i].name == "NormalizationLayer":
                    print(self.neuralNetwork.layer[i].output, np.sum(self.neuralNetwork.layer[i].output))
                    self.dLossdOutPrevBatch = np.zeros(shape=(self.batchSize, len(self.neuralNetwork.layer[i-1].node)))
                    if len(self.neuralNetwork.layer) - self.numberOfEndRescaleLayer == i:
                        self.numberOfEndRescaleLayer += 1
                    for batchIndex in range(self.batchSize):
                        #self.targetOutputBatch[batchIndex] is and array with 10 elements corresponding to the target output
                        self.yHat = self.targetOutputBatch[batchIndex]
                        #self.neuralNetwork.layer[i].output[batchIndex] is an array with 10 elements corresponding to the output of the layer
                        self.dLossdOutPrevBatch[batchIndex] = np.multiply(self.derivativeValue(lossFunction, self.neuralNetwork.layer[i].output[batchIndex]), 
                                                                            self.derivativeValue(self.neuralNetwork.layer[i].function, self.neuralNetwork.layer[i-1].output[batchIndex]))
                        if np.argmax(self.neuralNetwork.layer[i].output[batchIndex]) == np.argmax(self.targetOutputBatch[batchIndex]):
                            self.correct += 1
                            self.correctTotal += 1 
                        self.loss += np.sum(lossFunction(self.neuralNetwork.layer[i].output[batchIndex]))
                    self.dLossdOutBatch = self.dLossdOutPrevBatch
                else:
                    self.dLossdOutPrevBatch = np.zeros(shape=(self.batchSize, len(self.neuralNetwork.layer[i-1].node)))
                    for j in range(len(self.neuralNetwork.layer[i].node)-1, -1,-1):
                        self.batchWeightGrad = np.zeros(len(self.neuralNetwork.layer[i].node[j].weight))
                        self.batchBiasGrad = np.zeros(len(self.neuralNetwork.layer[i].node[j].bias))
                        for batchIndex in range(self.batchSize):
                            self.biasGrad = np.zeros(len(self.neuralNetwork.layer[i].node[j].bias))
                            self.weightGrad = np.zeros(len(self.neuralNetwork.layer[i].node[j].weight))
                            # if i == len(self.neuralNetwork.layer):
                            #     self.yHat = self.targetOutputBatch[batchIndex][j]
                            #     self.dLossdOut = self.derivativeValue(lossFunction, self.neuralNetwork.layer[i].node[j].outputBatch[batchIndex])
                            # else:
                            self.dLossdOut = self.dLossdOutBatch[batchIndex][j]
                            # self.neuralNetwork.layer[i].node[j].outputBatch[batchIndex] is a single node output from the batch
                            # self.neuralNetwork.layer[i].function is the activation function of the node in the layer
                            self.dOutdZ = self.derivativeValue(self.neuralNetwork.layer[i].function, self.neuralNetwork.layer[i].node[j].outputBatch[batchIndex])
                            # Next is to find the gradient of the loss with respect to the weights and bias                     
                            for k in range(len(self.neuralNetwork.layer[i].node[j].weight)):
                                # self.neuralNetwork.layer[i-1].node[k].outputBatch[batchIndex] is the output of the previous layer
                                self.dZdWeight = self.neuralNetwork.layer[i-1].node[k].outputBatch[batchIndex]
                                # print("dLossdOut: ", self.dLossdOut)
                                # print("dOutdZ: ", self.dOutdZ)
                                # print("dZdWeight: ", self.dZdWeight)
                                self.weightGrad[k] = self.dLossdOut * self.dOutdZ * self.dZdWeight
                                self.biasGrad[k] = self.dLossdOut * self.dOutdZ
                            #print("self.dLossdOutPrevBatch[batchIndex]: ", self.dLossdOutPrevBatch[batchIndex])
                            for k in range(len(self.neuralNetwork.layer[i].node[j].weight)):
                                self.dLossdOutPrevBatch[batchIndex][k] = np.add(self.dLossdOutPrevBatch[batchIndex][k], self.dLossdOut * self.dOutdZ * self.neuralNetwork.layer[i].node[j].weight[k])
                            self.batchWeightGrad = np.add(self.batchWeightGrad, self.weightGrad)
                            self.batchBiasGrad = np.add(self.batchBiasGrad, self.biasGrad)
                        self.batchWeightGrad = np.divide(self.batchWeightGrad, self.batchSize)
                        self.batchBiasGrad = np.divide(self.batchBiasGrad, self.batchSize)
                        self.neuralNetwork.layer[i].node[j].weight = np.subtract(self.neuralNetwork.layer[i].node[j].weight, self.learningRate * self.batchWeightGrad)
                        self.neuralNetwork.layer[i].node[j].bias = np.subtract(self.neuralNetwork.layer[i].node[j].bias, self.learningRate * self.batchBiasGrad)
                    self.dLossdOutBatch = self.dLossdOutPrevBatch
            self.correctHistory.append(self.correct/self.batchSize)
            self.accuracyHistory.append(self.correctTotal / (l+self.batchSize+1))
            print("Loss: " + str(self.loss))
            print("Correct: " + str(self.correct / self.batchSize))
            print("Accuracy: " + str(self.correctTotal / (l+self.batchSize+1)))
            with open('data.json', 'w') as f:
                json.dump({"correctHistory" : self.correctHistory,
                           "accuracyHistory" : self.accuracyHistory,
                           "correct" : self.correct / self.batchSize,
                            "accuracy" : self.correctTotal / (l+self.batchSize+1),
                           }, f)
                
                # if l%5000 == 0:
                #     #save weights and biases to json file in the structure neuralNetwork: {layer: {node: {weight: [], bias: []}}, ...}
                #     for i in range(1, len(self.neuralNetwork.layer)-1):
                #         layer = self.neuralNetwork.layer[i]
                #         layerDict = {}
                #         for j in range(len(layer.node)):
                #             node = layer.node[j]
                #             nodeDict = {"weight": node.weight.tolist(), "bias": node.bias.tolist()}
                #             layerDict[j] = nodeDict
                #         f.write(json.dumps({"neuralNetwork" : {"layer" : {i : layerDict}, "layerSize" : {i: len(layer.node)}}}))
                #     f.write(json.dumps({"learningRate" : self.learningRate, "batchSize" : self.batchSize}))

            # saveData = open("saveData.txt", "w")
            # saveData.write("Correct: " + str(self.correctHistory) + "\n")
            # saveData.write("Accuracy: " + str(self.accurarcyHistory) + "\n")
        

    def derivativeValue(self, function, value):
        if isinstance(value, (list, np.ndarray)):
            return np.divide(np.subtract(function(np.add(value, 0.0001)), function(value)), 0.0001)
        else:
            return (function(value + 0.0001) - function(value)) / 0.0001

  
def printArrayToAscii(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == 0:
                print("[0]", end = "")
            else:
                print("[1]", end = "")
        print()

import tensorflow_datasets as tfds
ds = tfds.load('mnist', split='train', shuffle_files=True)
ya = ds.take(30000)
# split the dataset with image and label
inputImageData = []
inputLabelData = []
for example in ya:
    labelArray = np.zeros(10)
    image, label = example["image"], example["label"]
    labelArray[label.numpy()] = 1
    inputImageData.append(image.numpy().flatten())
    inputLabelData.append(labelArray)
    #printArrayToAscii(image.numpy())
inputImageData = np.divide(inputImageData, 255)

batchSize = 5
NeuralNetwork = NeuralNetwork(InputLayer())
NeuralNetwork.addLayer(Layer(numberOfNodes=392, weightSize=784, biasSize=784, batchSize=batchSize, function=Functions.ReLU))
NeuralNetwork.addLayer(Layer(numberOfNodes=200, weightSize=392, biasSize=392, batchSize=batchSize, function=Functions.ReLU))
NeuralNetwork.addLayer(Layer(numberOfNodes=200, weightSize=200, biasSize=200, batchSize=batchSize, function=Functions.ReLU))
NeuralNetwork.addLayer(Layer(numberOfNodes=200, weightSize=200, biasSize=200, batchSize=batchSize, function=Functions.ReLU))
NeuralNetwork.addLayer(Layer(numberOfNodes=100, weightSize=200, biasSize=200, batchSize=batchSize, function=Functions.ReLU))
NeuralNetwork.addLayer(Layer(numberOfNodes=10, weightSize=100, biasSize=100, batchSize=batchSize, function=Functions.ReLU))
# NeuralNetwork.addLayer(Layer(numberOfNodes=392, weightSize=784, biasSize=784, batchSize=batchSize, function=Functions.ReLU))
# NeuralNetwork.addLayer(Layer(numberOfNodes=200, weightSize=392, biasSize=392, batchSize=batchSize, function=Functions.ReLU))
# NeuralNetwork.addLayer(Layer(numberOfNodes=200, weightSize=200, biasSize=200, batchSize=batchSize, function=Functions.ReLU))
# NeuralNetwork.addLayer(Layer(numberOfNodes=200, weightSize=200, biasSize=200, batchSize=batchSize, function=Functions.ReLU))
# NeuralNetwork.addLayer(Layer(numberOfNodes=10, weightSize=200, biasSize=200, batchSize=batchSize, function=Functions.ReLU))
NeuralNetwork.addLayer(NormalizationLayer())
NeuralNetwork.randomize()
BackPropagation = BackPropagation(NeuralNetwork, learningRate=0.1, batchSize=batchSize)
# load("Network", NeuralNetwork, BackPropagation)
BackPropagation.train(trainingData=inputImageData, trainingLabel=inputLabelData, lossFunction=BackPropagation.lossSquaredError)