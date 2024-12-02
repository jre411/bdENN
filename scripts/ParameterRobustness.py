'''
Path: bdENS/repo/scripts/ParameterRobustness.py

Boundary-Detecting ENN Project
bdENC Performance Evaluation
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 12-02-2024

Reference(s): 
    - bdENS/release/scripts/FigureScripts/Fig3e.py

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
subprojID = "ParameterRobustness"
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

# Variables
globers = {
    "randomSeed": 42,
    "nClasses": 3,
    "trainTestSplit": 0.85,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3,
    #"tol": 4e-1,
    "maxNoise": 1,
    "nSteps": 1000,
    "nTrials": 10000,
    "layerList": [1],
}

saveSuffix = "{}n{}s{}t-{}r".format(globers["maxNoise"], globers["nSteps"], globers["nTrials"], globers["randomSeed"])

# Load results
ennAccs = np.load(logMan.outputDir + "/ennAccs_{}.npy".format(saveSuffix))
mlpAccs = np.load(logMan.outputDir + "/mlpAccs_{}.npy".format(saveSuffix))

figWidth = 5
figHeight = 2.5
fig, ax = plt.subplots(1, 2, figsize=(figWidth, figHeight))

showQuartiles = True
ax[0].plot(np.median(ennAccs, axis=0), label="ENN", color=cC.networkColors["enn"])
ax[0].plot(np.median(mlpAccs, axis=0), label="Backprop", color=cC.networkColors["mlp"])
ax[0].set_xticks([0, 250, 500, 750, 1000], labels=["0", "0.25", "0.5", "0.75", "1"])

if showQuartiles:
    ax[0].fill_between(range(globers["nSteps"]), np.quantile(ennAccs, 0.25, axis=0), 
                    np.quantile(ennAccs, 0.75, axis=0), alpha=0.2, color=cC.networkColors["enn"])
    ax[0].fill_between(range(globers["nSteps"]), np.quantile(mlpAccs, 0.25, axis=0), 
                    np.quantile(mlpAccs, 0.75, axis=0), alpha=0.2, color=cC.networkColors["mlp"])
    
ax[0].set_xlabel("Fractional Noise")
ax[0].set_ylabel("Accuracy")

# Variables
globers = {
    "randomSeed": 42,
    "nClasses": 3,
    "trainTestSplit": 0.8,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3,
    "nTrials": 50
}

enners = {
    "subconceptSelector": 0,
}

mlpers = {
    "maxIter": 10000,
    "activation": "tanh",
}


xAll, yAll = cF.loadEmbeddings()
nFeatures = 512
accsENN = np.zeros((globers["nTrials"], nFeatures))

for i in range(globers["nTrials"]):
    xAll, yAll = cF.loadEmbeddings()
    xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, 
                                                    train_size=globers["trainTestSplit"], 
                                                    random_state=globers["randomSeed"]+i, 
                                                    stratify=yAll)
    
    # Train the ENN
    mdlENN = lB.main_0(xTrain, yTrain, enners["subconceptSelector"], globers["nHiddenNeurons"])

    inputWeights = mdlENN.layers[0].weights
    inputWeightsFrame = np.reshape(inputWeights, (4, 512, 3))
    inputWeightsFrameAvg = np.mean(np.abs(inputWeightsFrame), axis=0)
    inputWeightsFrameAvgAvg = np.mean(inputWeightsFrameAvg, axis=1)

    sortedIndices = np.argsort(inputWeightsFrameAvgAvg)

    for i2 in range(nFeatures):
        featureFocus = [sortedIndices[i2], sortedIndices[i2]+nFeatures, sortedIndices[i2]+2*nFeatures, sortedIndices[i2]+3*nFeatures]
        avgSignal = np.mean(xAll[:, featureFocus])
        xAll[:, featureFocus] = np.array([avgSignal, avgSignal, avgSignal, avgSignal]).T

        acc = 1 - mdlENN.compute_error(xAll, yAll)[0]
        accsENN[i, i2] = acc


accsMLP = np.zeros((globers["nTrials"], nFeatures))

for i in range(globers["nTrials"]):
    xAll, yAll = cF.loadEmbeddings()

    xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, 
                                                    train_size=globers["trainTestSplit"], 
                                                    random_state=globers["randomSeed"]+i, 
                                                    stratify=yAll)
    
    # Train the MLP
    mdlMLP = MLPClassifier(hidden_layer_sizes=(globers["nHiddenNeurons"], globers["nHiddenNeurons"]), 
                           max_iter=mlpers["maxIter"], random_state=globers["randomSeed"]+i, 
                           activation=mlpers["activation"])
    mdlMLP.fit(xTrain, yTrain)

    inputWeights = mdlMLP.coefs_[0]
    inputWeightsFrame = np.reshape(inputWeights, (4, nFeatures, 3))
    inputWeightsFrameAvg = np.mean(np.abs(inputWeightsFrame), axis=0)
    inputWeightsFrameAvgAvg = np.mean(inputWeightsFrameAvg, axis=1)

    sortedIndices = np.argsort(inputWeightsFrameAvgAvg)

    for i2 in range(nFeatures):

        featureFocus = [sortedIndices[i2], sortedIndices[i2]+nFeatures, sortedIndices[i2]+2*nFeatures, sortedIndices[i2]+3*nFeatures]
        avgSignal = np.mean(xAll[:, featureFocus])
        xAll[:, featureFocus] = np.array([avgSignal, avgSignal, avgSignal, avgSignal]).T

        acc = mdlMLP.score(xAll, yAll)
        accsMLP[i, i2] = acc

mlpMask = accsMLP[:,0] > 0.5
convergedMLPs = accsMLP[mlpMask]

ax[1].plot(np.mean(accsENN, axis=0), label="ENN", color=cC.networkColors["enn"])
ax[1].plot(np.mean(convergedMLPs, axis=0), label="Backprop", color=cC.networkColors["mlp"])

if showQuartiles:
    ax[1].fill_between(range(nFeatures), np.quantile(accsENN, 0.25, axis=0), 
                    np.quantile(accsENN, 0.75, axis=0), alpha=0.2, color=cC.networkColors["enn"])
    ax[1].fill_between(range(nFeatures), np.quantile(convergedMLPs, 0.25, axis=0), 
                    np.quantile(convergedMLPs, 0.75, axis=0), alpha=0.2, color=cC.networkColors["mlp"])

ax[0].set_yticks([0.5, 0.75, 1.0])
ax[1].set_yticks([0., 0.5, 1.0])
ax[1].set_xticks([0, 128, 256, 384, 512])
ax[1].set_xlabel("Features removed")
ax[1].set_ylabel("Accuracy")

fig.savefig(logMan.mediaDir + "/Fig3e.png",
            transparent=True, 
            dpi=300)
plt.close(fig)