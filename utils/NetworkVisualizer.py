# Path: release/utils/NetworkVisualizer.py

# Top-down learning is insufficient for cognitive modules

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# Network visualizer class
class Neuron():
    def __init__(self, x, y, radius=0.1, color="black"):
        """Class to represent a neuron in a network."""
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.connectionColor = "grey"
        self.connections = []
        self.alpha = 1
        self.lineWidth = 1
        self.borderColor = "black"
        self.connectionState = [0, 0, 0]

    def plotNeuron(self, _ax):
        """Plot the neuron."""
        circle = patches.Circle((self.x, self.y), self.radius, 
                                facecolor=self.color, fill=True, 
                                edgecolor=self.borderColor, linewidth=self.lineWidth,
                                alpha=self.alpha) 
        _ax.add_patch(circle)

    def plotConnections(self, _ax, _alpha=0.05):
        """Plot the connections of the neuron."""
        for _connection in self.connections:
            x = np.linspace(_connection[0][0], _connection[0][1], 100)
            xMapped = np.interp(x, (_connection[0][0], _connection[0][1]), (-6, 6))
            y = 1 / (1 + np.exp(-xMapped)) * (_connection[1][1] - _connection[1][0]) + _connection[1][0]

            # Make a sigmoid connection
            _ax.plot(x, y, color=self.connectionColor, alpha=_alpha, zorder=-1)
            _ax.plot(x, y+0.01, color="black", alpha=1, zorder=-1, linewidth=0.025)
            _ax.plot(x, y-0.01, color="black", alpha=1, zorder=-1, linewidth=0.025)
            


class NetworkVisualizer():
    def __init__(self, weights):
        """Class to visualize a network."""
        self.weights = weights
        self.neurons = []
        self.inputNeurons = []
        self.hiddenNeurons = [[] for i in range(len(self.weights)-1)]
        self.outputNeurons = []
        self.concepts = ["NB", "SB", "HB"]

        self.createNeurons()
        self.createConnections()

    def createNeurons(self):
        self.createInputNeurons()
        self.createHiddenNeurons()
        self.createOutputNeurons()

    def createInputNeurons(self):
        _nCols = 16
        _neuronsPerRow = self.weights[0].shape[0] / _nCols
        _nRows = self.weights[0].shape[0] // _neuronsPerRow
        _currCol = 0
        _currRow = 0
        for i in range(self.weights[0].shape[0]):
            if i % _neuronsPerRow == 0:
                _currCol += 1

            _currRow = i % _neuronsPerRow

            self.inputNeurons.append(Neuron(1.5*_currCol/_nCols - 1, 0.5 - 2*(_currRow/_neuronsPerRow), color="white", radius=0.025))

    def createHiddenNeurons(self):
        for i in range(self.hiddenNeurons.__len__()):
            for j in range(self.weights[i].shape[1]):
                self.hiddenNeurons[i].append(Neuron(1.25 + 1.25*i/(len(self.weights)-1), -j/(self.weights[i].shape[1]-1), color="white"))

    def createOutputNeurons(self):
        _conceptColors = ["#ACE7C5", "#92D6F0", "#D28383"]
        for i in range(3):
            self.outputNeurons.append(Neuron(2.75, 0.25-3*i/4, color=_conceptColors[i], radius=0.15))
            self.outputNeurons[i].concept = self.concepts[i]

    def createConnections(self):
        for i in range(self.inputNeurons.__len__()):
            for j in range(self.hiddenNeurons[0].__len__()):
                _line = [(self.inputNeurons[i].x, self.hiddenNeurons[0][j].x), (self.inputNeurons[i].y, self.hiddenNeurons[0][j].y)]
                self.inputNeurons[i].connections.append(_line)

        for i in range(self.hiddenNeurons[0].__len__()):
            for j in range(self.hiddenNeurons[1].__len__()):
                _line = [(self.hiddenNeurons[0][i].x, self.hiddenNeurons[1][j].x), (self.hiddenNeurons[0][i].y, self.hiddenNeurons[1][j].y)]
                self.hiddenNeurons[0][i].connections.append(_line)

        for i in range(self.hiddenNeurons[1].__len__()):
            for j in range(self.outputNeurons.__len__()):
                _line = [(self.hiddenNeurons[1][i].x, self.outputNeurons[j].x), (self.hiddenNeurons[1][i].y, self.outputNeurons[j].y)]
                self.hiddenNeurons[1][i].connections.append(_line)

    def plotNetwork(self, _ax):
        """Plot the network."""
        for _neuron in self.inputNeurons:
            _neuron.plotNeuron(_ax, _lineWidth=0.5)
            _neuron.plotConnections(_ax, _alpha=0.05)

        for _neuron in self.hiddenNeurons[0]:
            _neuron.plotNeuron(_ax)
            _neuron.plotConnections(_ax)

        for _neuron in self.hiddenNeurons[1]:
            _neuron.plotNeuron(_ax)
            _neuron.plotConnections(_ax)

        for _neuron in self.outputNeurons:
            _neuron.plotNeuron(_ax)
            _x, _y = _neuron.x, _neuron.y
            # Annotate the concept [0, 1, 2]
            _ax.annotate(_neuron.concept, (_x, _y), fontsize=10, ha="center", va="center", color="black")


        _ax.set_xlim(-1, 3)
        _ax.set_ylim(-2, 1)

        _ax.axis("off")

class CircuitVisualizer():
    def __init__(self):
        """Class to visualize a circuit."""
        self.inputNeurons = []
        self.boundaryNeurons = []
        self.eventNeurons = []
        self.connections = []

        self.createNeurons()
        self.createConnections()

    def createNeurons(self):
        self.createInputNeurons()
        self.createBoundaryNeurons()
        self.createEventNeurons()

    def createInputNeurons(self):
        _nInputNeurons = 50
        _nCols = 5
        _nRows = _nInputNeurons // _nCols

        _nFrames = 4
        for i2 in range(_nFrames):
            for i in range(_nInputNeurons):
                _x = i % _nCols * 0.25 - i2 * 0.02
                _y = i // _nCols * 0.25 - i2 * 0.02
                self.inputNeurons.append(Neuron(_x, _y, color="white", radius=0.075))
                self.inputNeurons[-1].lineWidth = 0.5
    
    def createBoundaryNeurons(self):
        _nBoundaryNeurons = 2

        for i in range(_nBoundaryNeurons):
            self.boundaryNeurons.append(Neuron(2, i+.5, color="#ffaac6", radius=0.25))

    def createEventNeurons(self):
        _nEventNeurons = 1

        for i in range(_nEventNeurons):
            self.eventNeurons.append(Neuron(3, i+1, color="#b85fd3", radius=0.25))

    def createConnections(self):
        for _inputNeuron in self.inputNeurons:
            for _boundaryNeuron in self.boundaryNeurons:
                _line = [(_inputNeuron.x, _boundaryNeuron.x), (_inputNeuron.y, _boundaryNeuron.y)]
                _boundaryNeuron.connections.append(_line)
                _boundaryNeuron.connectionState = [0, 0, 0]
            for _eventNeuron in self.eventNeurons:
                _line = [(_inputNeuron.x, _eventNeuron.x), (_inputNeuron.y, _eventNeuron.y)]
                _eventNeuron.connections.append(_line)



    def plotCircuit(self, _ax):
        """Plot the circuit."""
        for _neuron in self.inputNeurons:
            _neuron.plotNeuron(_ax)

        for _neuron in self.boundaryNeurons:
            _neuron.plotNeuron(_ax)
            _neuron.plotConnections(_ax, _alpha=0.05)

        for _neuron in self.eventNeurons:
            _neuron.plotConnections(_ax, _alpha=0.05)
            _neuron.plotNeuron(_ax)

        _ax.set_xlim(-.5, 3.5)
        _ax.set_ylim(-.5, 2.5)

        _ax.axis("off")