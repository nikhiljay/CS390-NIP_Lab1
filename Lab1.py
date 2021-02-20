
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
import time
from sklearn.metrics import confusion_matrix, f1_score
import pretty_errors

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Disable some troublesome logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return x * (1 - x)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 5, minibatches = True, mbs = 100):
        xVals = np.reshape(xVals, (60000, 784)) # flatten

        start_time = time.time()
        for iteration in range(epochs):
            xBatches = self.__batchGenerator(xVals, mbs) if minibatches else xVals
            yBatches = self.__batchGenerator(yVals, mbs) if minibatches else yVals
            
            for x, y in zip(xBatches, yBatches):
                layer1, layer2 = self.__forward(x)
                
                # Backward pass
                error = (layer2 - y) / self.outputSize * self.__sigmoidDerivative(layer2)
                dW2 = np.dot(layer1.T, error)
                error = np.dot(error, self.W2.T) * self.__sigmoidDerivative(layer1)
                dW1 = np.dot(x.T, error)

                # Update weights
                self.W1 -= self.lr * dW1
                self.W2 -= self.lr * dW2

            print('Epoch: {0}, Time Spent: {1:.2f}s'.format(iteration+1, time.time() - start_time))

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2

    # Classify test samples
    def classify(self, xTest):
        xTest = np.reshape(xTest, (10000, 784))
        ans = []
        for entry in xTest:
            max_val = np.argmax(self.predict(entry))
            pred = np.array([int(i == max_val) for i in range(10)])
            ans.append(pred)
        return np.array(ans)

class TensorFlow_2Layer():
    def __init__(self, neuronsPerLayer, learningRate = 0.1):
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(neuronsPerLayer, kernel_size=3, activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.Conv2D(neuronsPerLayer, kernel_size=3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def train(self, xVals, yVals, epochs = 1, minibatches = True, mbs = 100):
        xVals = np.reshape(xVals, (60000, 28, 28, 1))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        mbs = xVals if not minibatches else mbs
        self.model.fit(xVals, yVals, batch_size=mbs, epochs=epochs)

    def classify(self, xTest):
        xTest = np.reshape(xTest, (10000, 28, 28, 1))
        ans = []
        for entry in self.model.predict(xTest):
            max_val = np.argmax(entry)
            pred = np.array([int(i == max_val) for i in range(10)])
            ans.append(pred)
        return np.array(ans)

# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain = np.float32(xTrain) / 255
    xTest = np.float32(xTest) / 255
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))

def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        model = NeuralNetwork_2Layer(inputSize=784, outputSize=10, neuronsPerLayer=32, learningRate=0.5)
        model.train(xTrain, yTrain)
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = TensorFlow_2Layer(neuronsPerLayer=64, learningRate=0.01)
        model.train(xTrain, yTrain)
        return model
    else:
        raise ValueError("Algorithm not recognized.")

def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.classify(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        return model.classify(data)
    else:
        raise ValueError("Algorithm not recognized.")

def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Confusion matrix:")
    print(confusion_matrix(yTest.argmax(axis=1), preds.argmax(axis=1)))
    print("F1 Score: ", f1_score(yTest.argmax(axis=1), preds.argmax(axis=1), average='weighted'))
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):
            acc += 1
    accuracy = acc / preds.shape[0]
    print("Classifier accuracy: %f%%" % (accuracy * 100))

#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

if __name__ == '__main__':
    main()
