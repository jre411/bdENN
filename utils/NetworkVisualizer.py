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

        # Make neurons
        self.inputNeurons = []
        self.hiddenNeurons = [[] for _ in range(self.nHiddenLayers)]
        self.outputNeurons = []

    def createNeurons(self):
        """Create neurons for the network."""
        # Input neurons
        for i in range(self.inputDims):
            for j in range(1):
                _neuron = Neuron(i // self.nInputRows - j/10, i % self.nInputRows + j/10, 
                                color="white", radius=0.3)
                _neuron.lineWidth = 1
                self.inputNeurons.append(_neuron)

                
        # Hidden neurons
        _center = np.array([30, 7.5])
        for i in range(self.nHiddenLayers):
            for j in range(self.hiddenDims):
                _neuron = Neuron(_center[0], _center[1] + j*7.5, 
                                 color="white", radius=1.5)
                self.hiddenNeurons[i].append(_neuron)
            _center[0] += 15
        
        # Output neurons
        _center = np.array([60, 2.5])
        _conceptColors = ["#D28383", "#92D6F0", "#ACE7C5"]
        _concepts = ["HB", "SB", "NB"]
        for i in range(self.outputDims):
            _neuron = Neuron(_center[0], _center[1] + i*12.5, 
                                color=_conceptColors[i], radius=2.5)
            _neuron.concept = _concepts[i]
            self.outputNeurons.append(_neuron)

    def createConnections(self):
        """Create connections between each connected layer."""
        # Input to hidden
        for _ in self.inputNeurons:
            for __ in self.hiddenNeurons[0]:
                _.connections.append(
                    [(_.x, __.x), (_.y, __.y)]
                )

        # Hidden to hidden
        for _ in self.hiddenNeurons[0]:
            for __ in self.hiddenNeurons[1]:
                _.connections.append(
                    [(_.x, __.x), (_.y, __.y)]
                )

        # Hidden to output
        for _ in self.hiddenNeurons[1]:
            for __ in self.outputNeurons:
                _.connections.append(
                    [(_.x, __.x), (_.y, __.y)]
                )

    def plotNetwork(self, ax):
        """Plot the network."""
        # Input layer
        for _ in self.inputNeurons:
            _.plotNeuron(ax)
            _.plotConnections(ax, alpha=0.05)

        # Hidden layers
        for i in range(self.nHiddenLayers):
            for j in range(self.hiddenDims):
                self.hiddenNeurons[i][j].plotNeuron(ax)
                self.hiddenNeurons[i][j].plotConnections(ax, alpha=1)

        # Output layer
        for _ in self.outputNeurons:
            _.plotNeuron(ax)  

    def annotateNetwork(self, ax):
        """Annotate the network."""
        for _ in self.outputNeurons:
            ax.annotate(_.concept, (_.x, _.y), fontsize=14, ha="center", va="center", color="black")


# Neuron class
class Neuron():
    def __init__(self, _x, _y, color="white", radius=1):
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

    def plotNeuron(self, _ax):
        """Plot the neuron."""
        circle = patches.Circle((self.x, self.y), self.radius, 
                                facecolor=self.color, fill=True,   
                                edgecolor=self.borderColor, linewidth=self.lineWidth,
                                alpha=self.alpha) 
        _ax.add_patch(circle)

    def plotConnections(self, _ax, alpha=0.05):
        """Plot the connections of the neuron."""
        for _connection in self.connections:
            x = np.linspace(_connection[0][0], _connection[0][1], 100)
            xMapped = np.interp(x, (_connection[0][0], _connection[0][1]), (-6, 6))
            y = 1 / (1 + np.exp(-xMapped)) * (_connection[1][1] - _connection[1][0]) + _connection[1][0]

            # Make a sigmoid connection
            _ax.plot(x, y, color=self.connectionColor, alpha=alpha, zorder=-1)
            _ax.plot(x, y+0.1, color="black", alpha=alpha, zorder=-1, linewidth=0.025)
            _ax.plot(x, y-0.1, color="black", alpha=alpha, zorder=-1, linewidth=0.025)