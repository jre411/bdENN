'''
Path: bdENS/repo/utils/NetworkVisualizer.py

Boundary-Detecting ENN Project
Network Visualizer Utility
Author: James R. Elder
Institution: UTSW

DOO: 11-07-2024
LU: 11-07-2024

Reference: bdENS/release/utils/NetworkVisualizer.py

- Python 3.12.2
- bdENSenv
- 3.12.x-anaconda
- linux-gnu (BioHPC)
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

class NetworkVisualizer():
    def __init__(self, *args, **kwargs):
        """Class to visualize a network."""

        # Input layer
        self.inputDims = kwargs.get("inputDims", 1)
        self.nFrames = kwargs.get("nFrames", 1)
        
        # Hidden layers
        self.nHiddenLayers = kwargs.get("nHiddenLayers", 1)
        self.hiddenDims = kwargs.get("nHiddenNeurons", 1)
        
        # Output layer
        self.outputDims = kwargs.get("outputDims", 1)

        # Default plot settings
        self.nInputRows = self.inputDims

        # Initialize neuron lists for each layer (input, hidden, output)
        self.inputNeurons = []
        self.hiddenNeurons = [[] for _ in range(self.nHiddenLayers)]
        self.outputNeurons = []

        # Weights
        self.inputWeights = kwargs.get("inputWeights", None)
        self.hiddenLayerWeights = kwargs.get("hiddenLayerWeights", None)

    def createNeurons(self):
        """Create neurons for the network."""
        # Input neurons
        for i in range(self.inputDims):
            _neuron = Neuron(i // self.nInputRows, i % self.nInputRows, 
                            color="white", radius=0.3)
            if self.inputWeights is not None:
                _neuron.weights = self.inputWeights[i]
            self.inputNeurons.append(_neuron)

        # Hidden neurons
        _center = np.array([30, 7.5]) # Center of the first hidden layer, ideally set this algorithmically
        for i in range(self.nHiddenLayers):
            for j in range(self.hiddenDims):
                _neuron = Neuron(_center[0], _center[1] + j*7.5, 
                                 color="white", radius=1.5)
                if self.hiddenLayerWeights is not None:
                    _neuron.weights = self.hiddenLayerWeights[i][j]
                self.hiddenNeurons[i].append(_neuron)
            _center[0] += 15
        
        # Output neurons
        _center = np.array([60, 2.5]) # Center of the output layer, ideally set this algorithmically
        _conceptColors = ["#D28383", "#92D6F0", "#ACE7C5"]
        _concepts = ["HB", "SB", "NB"]
        for i in range(self.outputDims):
            _neuron = Neuron(_center[0], _center[1] + i*12.5, 
                                color=_conceptColors[i], radius=2.5)
            _neuron.concept = _concepts[i]
            self.outputNeurons.append(_neuron)

    def createConnections(self, colorGradient=None):
        """
        Create connections between each connected layer.
        Assumes a fully connected network, wherein each neuron in a layer is connected to all neurons in the next layer.
        """
        # Input to hidden
        for _ in self.inputNeurons:
            for i in range(self.hiddenDims):
                _connection = Connection(_, self.hiddenNeurons[0][i])
                #if _.weights is not None:
                #    _connection.weight = _.weights[i]
                _.connections.append(_connection)

        # Hidden to hidden (here we only have 2 hidden layers, ideally this should be done in a loop)
        for i in range(self.hiddenDims):
            for i2 in range(self.hiddenDims):
                _connection = Connection(self.hiddenNeurons[0][i], self.hiddenNeurons[1][i2])
                if self.hiddenNeurons[0][i].weights is not None:
                    _connection.weight = self.hiddenNeurons[0][i].weights[i2]
                    if colorGradient is not None:
                        _connection.color = colorGradient[50+int(50 * _connection.weight)]
                self.hiddenNeurons[0][i].connections.append(_connection)
        
        # Hidden to output
        for i in range(self.hiddenDims):
            for i2 in range(self.outputDims):
                _connection = Connection(self.hiddenNeurons[1][i], self.outputNeurons[i2])
                if self.hiddenNeurons[1][i].weights is not None:
                    _connection.weight = self.hiddenNeurons[1][i].weights[i2]
                    if colorGradient is not None:
                        _connection.color = colorGradient[50+int(50 * _connection.weight)]
                self.hiddenNeurons[1][i].connections.append(_connection)

    def plotNetwork(self, ax):
        """Plot the network."""
        # Input layer
        for _ in self.inputNeurons:
            _.plotNeuron(ax)
            _.plotConnections(ax, alpha=0.05)

        # Hidden layers
        for _ in [neuron for layer in self.hiddenNeurons for neuron in layer]:
            _.plotNeuron(ax)
            _.plotConnections(ax, alpha=1)

        # Output layer
        for _ in self.outputNeurons:
            _.plotNeuron(ax)  

    def annotateNetwork(self, ax):
        """Annotate the network."""
        for _ in self.outputNeurons:
            ax.annotate(_.concept, (_.x, _.y), fontsize=14, ha="center", va="center", color="black")


# Neuron class
class Neuron():
    def __init__(self, _x, _y, color="white", radius=1, weights=None):
        """Class to represent a neuron."""
        self.x = _x
        self.y = _y
        self.color = color
        self.radius = radius
        self.alpha = 1
        self.lineWidth = 1
        self.borderColor = "black"
        self.connections = []
        self.connectionColor = "black"
        self.weights = weights

    def plotNeuron(self, _ax):
        """Plot the neuron."""
        circle = patches.Circle((self.x, self.y), self.radius, 
                                facecolor=self.color, fill=True,   
                                edgecolor=self.borderColor, linewidth=self.lineWidth,
                                alpha=self.alpha) 
        _ax.add_patch(circle)

    def plotConnections(self, _ax, alpha=1):
        """Plot the connections of the neuron."""
        for _connection in self.connections:
            _connection.plotConnection(_ax, alpha=alpha)

class Connection():
    def __init__(self, _neuron1, _neuron2, weight=None):
        """Class to represent a connection between two neurons."""
        self.neuron1 = _neuron1
        self.neuron2 = _neuron2
        self.weight = weight 
        self.tol = 1e-4
        self.color = "black"

    def plotConnection(self, _ax, alpha=1):
        """Plot the connection."""
        x = np.linspace(self.neuron1.x, self.neuron2.x, 100)
        xMapped = np.interp(x, (self.neuron1.x, self.neuron2.x), (-6, 6))
        y = 1 / (1 + np.exp(-xMapped)) * (self.neuron2.y - self.neuron1.y) + self.neuron1.y

        if self.weight is not None:
            _margin = 0.5
            if np.abs(self.weight) < self.tol:
                print("Zero weight connection detected.")
                return
            if self.color is not None:
                _color = self.color
            else:
                _color = "blue" if self.weight > 0 else "red"
            _ax.fill_between(x, y-_margin, y+_margin, color=_color, alpha=alpha, zorder=-1)
            _ax.plot(x, y+_margin, color="black", alpha=alpha, zorder=-1)
            _ax.plot(x, y-_margin, color="black", alpha=alpha, zorder=-1)

        else:
            _margin = 0.01

            _ax.plot(x, y, color=self.color, alpha=alpha, zorder=-1)
            _ax.plot(x, y+_margin, color="black", alpha=0.05, zorder=-2)
            _ax.plot(x, y-_margin, color="black", alpha=0.05, zorder=-2)

class CircuitVisualizer():
    def __init__(self):
        """Class to visualize a circuit."""
        self.inputNeurons = []
        self.boundaryNeurons = []
        self.eventNeurons = []
        self.connections = []

    def setConnectionStates(self, _connectionState):
        self.boundaryNeurons[0].connectionState = _connectionState[0]
        self.boundaryNeurons[1].connectionState = _connectionState[1]
        self.eventNeurons[0].connectionState = _connectionState[2]
        self.updateNeurons()

    def updateNeurons(self):
        for _boundaryNeuron in self.boundaryNeurons:
            if _boundaryNeuron.connectionState == 1:
                _boundaryNeuron.color = "#ffaac6"
                for _connection in _boundaryNeuron.connections:
                    _connection.color = "#FFE617"
            else:
                _boundaryNeuron.color = "lightgray"
                for _connection in _boundaryNeuron.connections:
                    _connection.color = "lightgray"
        for _eventNeuron in self.eventNeurons:
            if _eventNeuron.connectionState == 1:
                _eventNeuron.color = "#b85fd3"
                for _connection in _eventNeuron.connections:
                    _connection.color = "#FFE617"
            else:
                _eventNeuron.color = "lightgray"
                for _connection in _eventNeuron.connections:
                    _connection.color = "lightgray"

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
                _connection = Connection(_inputNeuron, _boundaryNeuron)
                _boundaryNeuron.connections.append(_connection)
            for _eventNeuron in self.eventNeurons:
                _connection = Connection(_inputNeuron, _eventNeuron)
                _eventNeuron.connections.append(_connection)

    def plotCircuit(self, _ax):
        """Plot the circuit."""
        for _neuron in self.inputNeurons:
            _neuron.plotNeuron(_ax)

        for _neuron in self.boundaryNeurons:
            _neuron.plotNeuron(_ax)
            _neuron.plotConnections(_ax, alpha=0.05)

        for _neuron in self.eventNeurons:
            _neuron.plotConnections(_ax, alpha=0.05)
            _neuron.plotNeuron(_ax)

        _ax.set_xlim(-.5, 3.5)
        _ax.set_ylim(-.5, 2.5)

        _ax.axis("off")

    def annotateCircuit(self, _ax):
        """Annotate the circuit."""
        for _neuron in self.boundaryNeurons:
            _ax.annotate("B", (_neuron.x, _neuron.y - .05), fontsize=12, ha="center", va="center", color="black")
        for _neuron in self.eventNeurons:
            _ax.annotate("E", (_neuron.x, _neuron.y - .05), fontsize=12, ha="center", va="center", color="black")
            
