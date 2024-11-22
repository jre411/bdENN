'''
Path: bdENS/repo/scripts/NetworkVisualization.py

Boundary-Detecting ENN Project
Network Visualization
- Figure 1e: Top Model Visualization
Author: James R. Elder
Institution: UTSW

DOO: 11-07-2024
LU: 11-07-2024

Reference: bdENS/develop/FigureScripts/Fig1e.py

- Python 3.12.2
- bdENSenv
- 3.12.x-anaconda
- linux-gnu (BioHPC)
'''

# Project variables
user = "s181641"
projID = "bdENS"
labID = "Lin_lab"
labAffiliation = "greencenter"
projDir = "/work/{}/{}/{}/".format(labAffiliation, user, projID)

# Subproject variables
subprojID = "NetworkVisualization"
branch = "repo"

# System imports
import time
today = time.strftime("%m-%d-%Y")
import os
if os.getcwd() != projDir + branch:
    print("Changing to project directory")
    os.chdir(projDir + branch) # Change to the project directory
import sys
sys.path.append(projDir + branch) # Add the project directory to the system path

# 3rd party imports
import matplotlib.pyplot as plt

# ENN imports

# Local imports
from utils import LumberJack as LJ
from utils import NetworkVisualizer as NV
from utils import commonFunctions as cF

# Logging
logMan = LJ.LoggingManager(subprojID)
cF.updatePlotrcs(font="serif")

# Variables
globers = {
    "randomSeed": 42,
    "nClasses": 3,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3,
    "nFrames": 4,
    "dimsPerFrame": 512
}

# Plot a representative network (Fig 1e)
figWidth = 6
figHeight = 3
fig, ax = plt.subplots(1, 1, figsize=(figWidth, figHeight))

# Initialize the network visualizer
networkVisualizer = NV.NetworkVisualizer(inputDims=globers["dimsPerFrame"], 
                               nFrames=globers["nFrames"],
                               nHiddenLayers=globers["nHiddenLayers"], 
                               nHiddenNeurons=globers["nHiddenNeurons"],
                               outputDims=globers["nClasses"], 
                               )

# Set plot parameters
networkVisualizer.nInputRows = 32

# Create the neurons and connections
networkVisualizer.createNeurons()
networkVisualizer.createConnections()

# Plot the network
networkVisualizer.plotNetwork(ax=ax)
networkVisualizer.annotateNetwork(ax=ax)

# Figure settings
ax.set_xlim(-1, networkVisualizer.nInputRows * 2 + 1)
ax.set_ylim(-1, networkVisualizer.nInputRows + 1)
ax.axis("off")
fig.tight_layout()

# Save the figure
fig.savefig(logMan.mediaDir + "/Fig1e.png",
            transparent=True, 
            dpi=300)
plt.close(fig)

# Plot a representative ENN (Fig 3c)
initialColorHex = "#808080" # Gray
middleColorHex = "#FFFFFF" # White
finalColorHex = "#ffe116" # Gold

# Generate the color gradient
colorGradient1 = cF.genColorGradient(initialColorHex, middleColorHex, 50)
colorGradient2 = cF.genColorGradient(middleColorHex, finalColorHex, 51)
colorGradient = colorGradient1 + colorGradient2

# Load the embeddings
import numpy as np
from sklearn.model_selection import train_test_split
xAll, yAll = cF.loadEmbeddings()
xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, test_size=0.15, random_state=42, stratify=yAll)

import enn.learnBoundaries as lB
mdlENN = lB.main_0(xTrain, yTrain, 0, globers["nHiddenNeurons"]) # Train the ENN

# Get the weights
enn_InputWeights = mdlENN.layers[0].weights
enn_HL1Weights = mdlENN.layers[1].weights
enn_HL2Weights = mdlENN.layers[2].weights

# Reshape input weights to get the frame absolute average, normalized
enn_InputWeights512 = np.reshape(enn_InputWeights, (globers["nFrames"], 
                                                    globers["dimsPerFrame"], 
                                                    globers["nHiddenNeurons"]))
enn_InputWeights512AbsAvg = np.mean(np.abs(enn_InputWeights512), axis=0)
enn_InputWeights512AbsAvgNorm = enn_InputWeights512AbsAvg / np.max(np.abs(enn_InputWeights512))

# Normalize the hidden layer weights
enn_HL1WeightsNorm = enn_HL1Weights / np.max(np.abs(enn_HL1Weights))
enn_HL2WeightsNorm = enn_HL2Weights / np.max(np.abs(enn_HL2Weights))

figWidth = 6
figHeight = 3
fig, ax = plt.subplots(1, 1, figsize=(figWidth, figHeight))

# Initialize the network visualizer
networkVisualizer = NV.NetworkVisualizer(inputDims=globers["dimsPerFrame"], 
                               nFrames=globers["nFrames"],
                               nHiddenLayers=globers["nHiddenLayers"], 
                               nHiddenNeurons=globers["nHiddenNeurons"],
                               outputDims=globers["nClasses"], 
                               inputWeights=enn_InputWeights512AbsAvgNorm,
                               hiddenLayerWeights=[enn_HL1WeightsNorm, enn_HL2WeightsNorm],                        
                               )

# Set plot parameters
networkVisualizer.nInputRows = 32

# Create the neurons and connections
networkVisualizer.createNeurons()
networkVisualizer.createConnections(colorGradient)

# Plot the network
networkVisualizer.plotNetwork(ax=ax)
networkVisualizer.annotateNetwork(ax=ax)

# Figure settings
ax.set_xlim(-1, networkVisualizer.nInputRows * 2 + 1)
ax.set_ylim(-1, networkVisualizer.nInputRows + 1)
ax.axis("off")
fig.tight_layout()

# Save the figure
fig.savefig(logMan.mediaDir + "/Fig3c.png",
            transparent=True, 
            dpi=300)
plt.close(fig)

# Plot a representative MLP (Fig 3b)
from sklearn.neural_network import MLPClassifier

mdlMLP = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=1000, random_state=42)
mdlMLP.fit(xTrain, yTrain)

# Get the weights
mlp_InputWeights = mdlMLP.coefs_[0]
mlp_HL1Weights = mdlMLP.coefs_[1]
mlp_HL2Weights = mdlMLP.coefs_[2]

# Reshape input weights to get the frame absolute average, normalized
mlp_InputWeights512 = np.reshape(mlp_InputWeights, (globers["nFrames"], 
                                                    globers["dimsPerFrame"], 
                                                    globers["nHiddenNeurons"]))
mlp_InputWeights512AbsAvg = np.mean(np.abs(mlp_InputWeights512), axis=0)
mlp_InputWeights512AbsAvgNorm = mlp_InputWeights512AbsAvg / np.max(np.abs(mlp_InputWeights512))

# Normalize the hidden layer weights
mlp_HL1WeightsNorm = mlp_HL1Weights / np.max(np.abs(mlp_HL1Weights))
mlp_HL2WeightsNorm = mlp_HL2Weights / np.max(np.abs(mlp_HL2Weights))

figWidth = 6
figHeight = 3
fig, ax = plt.subplots(1, 1, figsize=(figWidth, figHeight))

# Initialize the network visualizer
networkVisualizer = NV.NetworkVisualizer(inputDims=globers["dimsPerFrame"], 
                               nFrames=globers["nFrames"],
                               nHiddenLayers=globers["nHiddenLayers"], 
                               nHiddenNeurons=globers["nHiddenNeurons"],
                               outputDims=globers["nClasses"], 
                               inputWeights=mlp_InputWeights512AbsAvgNorm,
                               hiddenLayerWeights=[mlp_HL1WeightsNorm, mlp_HL2WeightsNorm],                        
                               )

# Set plot parameters
networkVisualizer.nInputRows = 32

# Create the neurons and connections
networkVisualizer.createNeurons()
networkVisualizer.createConnections(colorGradient)

# Plot the network
networkVisualizer.plotNetwork(ax=ax)
networkVisualizer.annotateNetwork(ax=ax)

# Figure settings
ax.set_xlim(-1, networkVisualizer.nInputRows * 2 + 1)
ax.set_ylim(-1, networkVisualizer.nInputRows + 1)
ax.axis("off")
fig.tight_layout()

# Save the figure
fig.savefig(logMan.mediaDir + "/Fig3b.png",
            transparent=True, 
            dpi=300)
plt.close(fig)
