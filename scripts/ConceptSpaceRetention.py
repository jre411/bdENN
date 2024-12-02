'''
Path: bdENS/repo/scripts/ConceptSpaceRetention.py

Boundary-Detecting ENN Project
Concept Space Retention
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 12-02-2024

Reference(s): 
    - bdENS/release/FigureScripts/Fig3f.py

- Python 3.12.2
- bdENSenv
- 3.12.x-anaconda
- linux-gnu (BioHPC)
'''

user = "s181641"
projID = "bdENS"
labID = "Lin_lab"
labAffiliation = "greencenter"
projDir = "/work/{}/{}/{}/".format(labAffiliation, user, projID)

# Subproject variables
subprojID = "ConceptSpaceRetention"
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
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from copy import deepcopy

# ENN imports
from enn.network import Network
import enn.learnBoundaries as lB

# Local imports
from utils import LumberJack as LJ
from utils import commonFunctions as cF
from utils import commonClasses as cC
from utils import NetworkVisualizer as NV

# Logging
logMan = LJ.LoggingManager(subprojID)
cF.updatePlotrcs(font="serif")

globers = {
    "randomSeed": 42,
    "classes": ["No Boundary", "Soft Boundary", "Hard Boundary"],
    "trainTestSplit": 0.85,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3
}

mlpers = {
    "maxIter": 10000,
    "activation": "tanh"
}

enners = {
    "subconeptSelector": 0,
}


xAll, yAll = cF.loadEmbeddings()
xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, 
                                                train_size=globers["trainTestSplit"],
                                                random_state=globers["randomSeed"], 
                                                stratify=yAll)

figWidth = 6
figHeight = 7
fig, ax = plt.subplots(3, 2, figsize=(figWidth, figHeight), subplot_kw={'projection': '3d'})

# Train base models
mdlENN = lB.main_0(xTrain, yTrain, enners["subconeptSelector"], globers["nHiddenNeurons"])
mdlMLP = MLPClassifier(hidden_layer_sizes=(globers["nHiddenNeurons"], globers["nHiddenNeurons"]), 
                       max_iter=mlpers["maxIter"], 
                       random_state=globers["randomSeed"], 
                       activation=mlpers["activation"])
mdlMLP.fit(xTrain, yTrain)

# Add in the training data
enn_l1Outputs = mdlENN.layers[0].compute_output(xTrain, activate=True) # Tanh activation
enn_l2Outputs = mdlENN.layers[1].compute_output(enn_l1Outputs, activate=True) # Tanh activation

mlp_l1Outputs_unAct = mdlMLP.coefs_[0].T @ xTrain.T + mdlMLP.intercepts_[0].reshape(-1, 1)
mlp_l1Outputs = np.tanh(mlp_l1Outputs_unAct) # Tanh activation
mlp_l2Outputs_unAct = mdlMLP.coefs_[1].T @ mlp_l1Outputs + mdlMLP.intercepts_[1].reshape(-1, 1)
mlp_l2Outputs = np.tanh(mlp_l2Outputs_unAct) # Tanh activation

class ScatterMarker:
    def __init__(self, marker, edgecolors):
        self.marker = marker
        self.edgecolors = edgecolors

    def add(self, _ax):
        _ax.scatter(marker=self.marker, edgecolors=self.edgecolors)

for i in range(len(enn_l2Outputs)):
    _marker = "s"
    ax[0,0].scatter(enn_l2Outputs[i][0], enn_l2Outputs[i][1], enn_l2Outputs[i][2], 
                 color=cC.conceptColors[yTrain[i]], alpha=1, 
                 marker=_marker, 
                 edgecolors="black", 
                 linewidth=1,
                 zorder=10)   
    
    ax[1,0].scatter(enn_l2Outputs[i][0], enn_l2Outputs[i][1], enn_l2Outputs[i][2], 
                 color=cC.conceptColors[yTrain[i]], alpha=1, 
                 marker=_marker, 
                 edgecolors="black", 
                 linewidth=1,
                 zorder=10)   

    ax[2,0].scatter(enn_l2Outputs[i][0], enn_l2Outputs[i][1], enn_l2Outputs[i][2], 
                 color=cC.conceptColors[yTrain[i]], alpha=1, 
                 marker=_marker, edgecolors="black", 
                 linewidth=1,
                 zorder=10)      
    
    ax[0,1].scatter(mlp_l2Outputs[0,i], mlp_l2Outputs[1,i], mlp_l2Outputs[2,i], 
                 color=cC.conceptColors[yTrain[i]], alpha=1, marker=_marker, edgecolors="black", 
                 linewidth=1,
                 zorder=10)   
    ax[1,1].scatter(mlp_l2Outputs[0,i], mlp_l2Outputs[1,i], mlp_l2Outputs[2,i],
                 color=cC.conceptColors[yTrain[i]], alpha=1, marker=_marker, edgecolors="black", 
                 linewidth=1,
                 zorder=10)   
    ax[2,1].scatter(mlp_l2Outputs[0,i], mlp_l2Outputs[1,i], mlp_l2Outputs[2,i],
                 color=cC.conceptColors[yTrain[i]], alpha=1, marker=_marker, edgecolors="black", 
                 linewidth=1,
                 zorder=10)   


# Get TP
ennTPs = cF.getTruePosInds(mdlENN, xTest, yTest, mdlType="enn")
mlpTPs = cF.getTruePosInds(mdlMLP, xTest, yTest, mdlType="mlp")

enn_l1Outputs = mdlENN.layers[0].compute_output(xTest[ennTPs], activate=True) # Tanh activation
enn_l2Outputs = mdlENN.layers[1].compute_output(enn_l1Outputs, activate=True) # Tanh activation

mlp_l1Outputs_unAct = mdlMLP.coefs_[0].T @ xTest[mlpTPs].T + mdlMLP.intercepts_[0].reshape(-1, 1)
mlp_l1Outputs = np.tanh(mlp_l1Outputs_unAct) # Tanh activation
mlp_l2Outputs_unAct = mdlMLP.coefs_[1].T @ mlp_l1Outputs + mdlMLP.intercepts_[1].reshape(-1, 1)
mlp_l2Outputs = np.tanh(mlp_l2Outputs_unAct) # Tanh activation

for i in range(len(enn_l2Outputs)):
    _marker = "o"
    ax[0,0].scatter(enn_l2Outputs[i][0], enn_l2Outputs[i][1], enn_l2Outputs[i][2], 
                 color=cC.conceptColors[yTest[ennTPs[i]]], alpha=1, marker=_marker, edgecolors="grey")

for i in range(len(mlp_l2Outputs)):
    _marker = "o"

    ax[0,1].scatter(mlp_l2Outputs[0,i], mlp_l2Outputs[1,i], mlp_l2Outputs[2,i], 
                 color=cC.conceptColors[yTest[mlpTPs[i]]], alpha=1, marker=_marker, edgecolors="grey")

noise = 0.2

for i2 in range(10):
    # Copy OG models
    noisyENN = mdlENN.copy()
    noisyMLP = deepcopy(mdlMLP) 

    # Inject noise
    noisyENN.layers[1].weights = mdlENN.layers[1].weights + mdlENN.layers[1].weights * noise * np.random.normal(size=mdlENN.layers[1].weights.shape)
    noisyMLP.coefs_[1] = mdlMLP.coefs_[1] + mdlMLP.coefs_[1] * noise * np.random.normal(size=mdlMLP.coefs_[1].shape)

    # Add in the test data
    enn_l1Outputs = noisyENN.layers[0].compute_output(xTest[ennTPs], activate=True) # Tanh activation
    enn_l2Outputs = noisyENN.layers[1].compute_output(enn_l1Outputs, activate=True) # Tanh activation

    mlp_l1Outputs_unAct = noisyMLP.coefs_[0].T @ xTest[mlpTPs].T + noisyMLP.intercepts_[0].reshape(-1, 1)
    mlp_l1Outputs = np.tanh(mlp_l1Outputs_unAct) # Tanh activation
    mlp_l2Outputs_unAct = noisyMLP.coefs_[1].T @ mlp_l1Outputs + noisyMLP.intercepts_[1].reshape(-1, 1)
    mlp_l2Outputs = np.tanh(mlp_l2Outputs_unAct) # Tanh activation

    for i in range(len(ennTPs)):
        _marker = "x"
        _alpha = 1
        if np.argmax(noisyENN.compute_output(xTest[ennTPs[i]])) == yTest[ennTPs[i]]:
            _marker = "o"
            _alpha = 0.5
        ax[1,0].scatter(enn_l2Outputs[i][0], enn_l2Outputs[i][1], enn_l2Outputs[i][2], 
                    color=cC.conceptColors[yTest[ennTPs[i]]], alpha=_alpha, marker=_marker, zorder=-1)
        
    for i in range(len(mlpTPs)):
        _marker = "x"
        _alpha = 1
        if noisyMLP.predict(xTest[mlpTPs[i]].reshape(1, -1)) == yTest[mlpTPs[i]]:
            _marker = "o"
            _alpha = 0.5
        ax[1,1].scatter(mlp_l2Outputs[0,i], mlp_l2Outputs[1,i], mlp_l2Outputs[2,i], 
                    color=cC.conceptColors[yTest[mlpTPs[i]]], alpha=_alpha, marker=_marker, zorder=-1)
        

noise = 0.8

for i2 in range(10):
    # Copy OG models
    noisyENN = mdlENN.copy()
    noisyMLP = deepcopy(mdlMLP) 

    # Inject noise
    noisyENN.layers[1].weights = mdlENN.layers[1].weights + mdlENN.layers[1].weights * noise * np.random.normal(size=mdlENN.layers[1].weights.shape)
    noisyMLP.coefs_[1] = mdlMLP.coefs_[1] + mdlMLP.coefs_[1] * noise * np.random.normal(size=mdlMLP.coefs_[1].shape)

    # Add in the test data
    enn_l1Outputs = noisyENN.layers[0].compute_output(xTest[ennTPs], activate=True) # Tanh activation
    enn_l2Outputs = noisyENN.layers[1].compute_output(enn_l1Outputs, activate=True) # Tanh activation

    mlp_l1Outputs_unAct = noisyMLP.coefs_[0].T @ xTest[mlpTPs].T + noisyMLP.intercepts_[0].reshape(-1, 1)
    mlp_l1Outputs = np.tanh(mlp_l1Outputs_unAct) # Tanh activation
    mlp_l2Outputs_unAct = noisyMLP.coefs_[1].T @ mlp_l1Outputs + noisyMLP.intercepts_[1].reshape(-1, 1)
    mlp_l2Outputs = np.tanh(mlp_l2Outputs_unAct) # Tanh activation

    for i in range(len(ennTPs)):
        _marker = "x"
        _alpha = 1
        if np.argmax(noisyENN.compute_output(xTest[ennTPs[i]])) == yTest[ennTPs[i]]:
            _marker = "o"
            _alpha = 0.5
        ax[2,0].scatter(enn_l2Outputs[i][0], enn_l2Outputs[i][1], enn_l2Outputs[i][2], 
                    color=cC.conceptColors[yTest[ennTPs[i]]], alpha=_alpha, marker=_marker, zorder=-1)
        
    for i in range(len(mlpTPs)):
        _marker = "x"
        _alpha = 1
        if noisyMLP.predict(xTest[mlpTPs[i]].reshape(1, -1)) == yTest[mlpTPs[i]]:
            _marker = "o"
            _alpha = 0.5
        ax[2,1].scatter(mlp_l2Outputs[0,i], mlp_l2Outputs[1,i], mlp_l2Outputs[2,i], 
                    color=cC.conceptColors[yTest[mlpTPs[i]]], alpha=_alpha, marker=_marker, zorder=-1)
        
# Add axes labels
for i in range(3):
    for j in range(2):
        ax[i, j].set_xlabel("Neuron 1")
        ax[i, j].set_ylabel("Neuron 2")
        ax[i, j].set_zlabel("Neuron 3")

# Plot the decision boundaries
nSteps = 20
samplers = np.linspace(-1, 1, nSteps)
samplerAlpha = 0.025

for i in samplers:
    for j in samplers:
        for k in samplers:
            x = np.array([i, j, k])

            y = np.argmax(mdlENN.layers[-1].compute_output(x))    
            ax[0,0].scatter(i, j, k, color=cC.conceptColors[y], alpha=samplerAlpha)
            ax[1,0].scatter(i, j, k, color=cC.conceptColors[y], alpha=samplerAlpha)
            ax[2,0].scatter(i, j, k, color=cC.conceptColors[y], alpha=samplerAlpha)

            y = np.argmax(mdlMLP.coefs_[2].T @ x + mdlMLP.intercepts_[2])
            ax[0,1].scatter(i, j, k, color=cC.conceptColors[y], alpha=samplerAlpha)
            ax[1,1].scatter(i, j, k, color=cC.conceptColors[y], alpha=samplerAlpha)
            ax[2,1].scatter(i, j, k, color=cC.conceptColors[y], alpha=samplerAlpha)

# Set ticks
for i in range(3):
    for j in range(2):
        ax[i, j].set_xticks([-1, 0, 1])
        ax[i, j].set_yticks([-1, 0, 1])
        ax[i, j].set_zticks([-1, 0, 1])

# Rotate the viewpoint of the first column
for i in range(3):
    ax[i, 0].view_init(elev=20, azim=45)
    ax[i, 1].view_init(elev=20, azim=-45)

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/Fig3f.png",
            transparent=True, 
            dpi=300)
plt.close(fig)