'''
Path: bdENS/repo/scripts/VariableSplitsAndReproducibility.py

Boundary-Detecting ENN Project
Variable Splits and Reproducibility
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 01-16-2025

Reference(s): 
    - bdENS/develop/scripts/LearningSolution.py

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
subprojID = "VariableSplitsAndReproducibility"
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

# Variables
globers = {
    "randomSeed": 9,
    "nClasses": 3,
    "trainTestSplit": 0.2,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3,
}

enners = {
    "subconceptSelector": 0,  
    "cPlot": "#800080",
    "csPlot": ["#4B0082", "#A020F0", "#DDA0DD"]
}

mlpers = {
    "maxIter": 10000,
    "activation": "tanh",
    "cPlot": "#d1c73d",
    "csPlot": ["#b2a81f", "#e3e04a", "#c4d75d"]
}

# Logging
logMan = LJ.LoggingManager(subprojID)
cF.updatePlotrcs(font="serif")

ennl1weights = np.load(logMan.outputDir + "/l1weightsENN.npy")
ennl2weights = np.load(logMan.outputDir + "/l2weightsENN.npy")
ennl3weights = np.load(logMan.outputDir + "/l3weightsENN.npy")

mlpl1weights = np.load(logMan.outputDir + "/l1weightsMLP.npy")
mlpl2weights = np.load(logMan.outputDir + "/l2weightsMLP.npy")
mlpl3weights = np.load(logMan.outputDir + "/l3weightsMLP.npy")

accsMLP = np.load(logMan.outputDir + "/accsMLP.npy")
accsENN = np.load(logMan.outputDir + "/accsENN.npy")

nSplits = accsENN.shape[0]
print("Number of splits: {}".format(nSplits))

threshold = 0.4
trialsBelowThreshold = []

mlpl1weightsAboveThreshold = np.array([])
ennl1weightsAboveThreshold = np.array([])

ennl2weightsAboveThreshold = np.array([])
ennl3weightsAboveThreshold = np.array([])

mlpl2weightsAboveThreshold = np.array([])
mlpl3weightsAboveThreshold = np.array([])

ennl1weightsAbsAvgAboveThreshold = np.array([])
mlpl1weightsAbsAvgAboveThreshold = np.array([])
    
nSplits = 1000
for i in range(nSplits):
    if accsMLP[i] > threshold:
        
        mlpl1weightsAbsAvg = np.mean(np.abs(mlpl1weights[i]), axis=1)
        mlpl1weightsAbsAvgAboveThreshold = np.append(mlpl1weightsAbsAvgAboveThreshold, mlpl1weightsAbsAvg)

        mlpl1weightsAboveThreshold = np.append(mlpl1weightsAboveThreshold, mlpl1weights[i])
        mlpl2weightsAboveThreshold = np.append(mlpl2weightsAboveThreshold, mlpl2weights[i])
        mlpl3weightsAboveThreshold = np.append(mlpl3weightsAboveThreshold, mlpl3weights[i])

        ennl1weightsAbsAvg = np.mean(np.abs(ennl1weights[i]), axis=1)
        ennl1weightsAbsAvgAboveThreshold = np.append(ennl1weightsAbsAvgAboveThreshold, ennl1weightsAbsAvg)

        ennl1weightsAboveThreshold = np.append(ennl1weightsAboveThreshold, ennl1weights[i])
        ennl2weightsAboveThreshold = np.append(ennl2weightsAboveThreshold, ennl2weights[i])
        ennl3weightsAboveThreshold = np.append(ennl3weightsAboveThreshold, ennl3weights[i])

    if i % 1000 == 0:
        print(i)

# Reshape weights
mlpl1weightsAbsAvgAboveThreshold = mlpl1weightsAbsAvgAboveThreshold.reshape(-1, mlpl1weights.shape[1])
mlpl1weightsAboveThreshold = mlpl1weightsAboveThreshold.reshape(-1, mlpl1weights.shape[1])
mlpl2weightsAboveThreshold = mlpl2weightsAboveThreshold.reshape(-1, mlpl2weights.shape[1])
mlpl3weightsAboveThreshold = mlpl3weightsAboveThreshold.reshape(-1, mlpl3weights.shape[1])

ennl1weightsAbsAvgAboveThreshold = ennl1weightsAbsAvgAboveThreshold.reshape(-1, ennl1weights.shape[1])
ennl1weightsAboveThreshold = ennl1weightsAboveThreshold.reshape(-1, ennl1weights.shape[1])
ennl2weightsAboveThreshold = ennl2weightsAboveThreshold.reshape(-1, ennl2weights.shape[1])
ennl3weightsAboveThreshold = ennl3weightsAboveThreshold.reshape(-1, ennl3weights.shape[1])

# Normalize Weights
ennl1weightsNormAbsAvg = ennl1weightsAbsAvgAboveThreshold / np.max(np.abs(ennl1weightsAbsAvgAboveThreshold))
ennl1weightsNorm = ennl1weightsAboveThreshold / np.max(np.abs(ennl1weightsAboveThreshold))
ennl2weightsNorm = ennl2weightsAboveThreshold / np.max(np.abs(ennl2weightsAboveThreshold))
ennl3weightsNorm = ennl3weightsAboveThreshold / np.max(np.abs(ennl3weightsAboveThreshold))

mlpl1weightsNormAbsAvg = mlpl1weightsAbsAvgAboveThreshold / np.max(np.abs(mlpl1weightsAbsAvgAboveThreshold))
mlpl1weightsNorm = mlpl1weightsAboveThreshold / np.max(np.abs(mlpl1weightsAboveThreshold))
mlpl2weightsNorm = mlpl2weightsAboveThreshold / np.max(np.abs(mlpl2weightsAboveThreshold))
mlpl3weightsNorm = mlpl3weightsAboveThreshold / np.max(np.abs(mlpl3weightsAboveThreshold))

xAll, yAll = cF.loadEmbeddings()

minSplit = 0.01
maxSplit = 1
nSplits = 10
splitList = np.linspace(minSplit, maxSplit, nSplits)
splitList = np.round(splitList, 2)

nSeeds = 1  
mlpCoefs = np.zeros((nSeeds, nSplits, globers["nHiddenLayers"], globers["nHiddenNeurons"], globers["nHiddenNeurons"]))
ennCoefs = np.zeros((nSeeds, nSplits, globers["nHiddenLayers"], globers["nHiddenNeurons"], globers["nHiddenNeurons"]))
mlpInputCoefs = np.zeros((nSeeds, nSplits, 512, globers["nHiddenNeurons"]))
ennInputCoefs = np.zeros((nSeeds, nSplits, 512, globers["nHiddenNeurons"]))

for i in range(nSeeds):
    mdlMLP = MLPClassifier(hidden_layer_sizes=(globers["nHiddenNeurons"], globers["nHiddenNeurons"]), 
                           max_iter=mlpers["maxIter"], random_state=globers["randomSeed"], 
                           activation="tanh")
    
    for i2, split in enumerate(splitList):
        if split != 1:
            xTrain, _, yTrain, _ = train_test_split(xAll, yAll, train_size=split, random_state=globers["randomSeed"], stratify=yAll)
        else:
            xTrain, yTrain = xAll, yAll
        print("Train: {}".format(len(xTrain)))
        mdlMLP.fit(xTrain, yTrain)
        mdlENN = lB.main_0(xTrain, yTrain, 0, globers["nHiddenNeurons"])
        mlpCoefs[i][i2][0] = mdlMLP.coefs_[1]
        mlpCoefs[i][i2][1] = mdlMLP.coefs_[2]

        ennCoefs[i][i2][0] = mdlENN.layers[1].weights
        ennCoefs[i][i2][1] = mdlENN.layers[2].weights

        mlpInputCoefs[i][i2] = np.mean(np.abs(np.reshape(mdlMLP.coefs_[0], (4, 512, 3))), axis=0)
        ennInputCoefs[i][i2] = np.mean(np.abs(np.reshape(mdlENN.layers[0].weights, (4, 512, 3))), axis=0)

    print("Seed {} done".format(i))
    
hlBins = np.linspace(-1,1,10)
inputBins = np.linspace(0,1,20)
startHex = "#000000"
endHex = "#d1c73d"
ennGradient = cF.genColorGradient(startHex, cC.networkColors["enn"], nSplits)
mlpGradient = cF.genColorGradient(startHex, cC.networkColors["mlp"], nSplits)

figWidth = 5
figHeight = 5
fig, ax = plt.subplots(4, 3, figsize=((figWidth, figHeight)))

ax[3,0].hist(ennl1weightsNormAbsAvg.flatten(), bins=inputBins, edgecolor='black', color=enners["cPlot"], label="ENN")
ax[3,1].hist(ennl2weightsNorm.flatten(), bins=hlBins, edgecolor='black', color=enners["cPlot"], label="ENN")
ax[3,2].hist(ennl3weightsNorm.flatten(), bins=hlBins, edgecolor='black', color=enners["cPlot"], label="ENN")

ax[1,0].hist(mlpl1weightsNormAbsAvg.flatten(), bins=inputBins, edgecolor='black', color=mlpers["cPlot"], label="Backprop")
ax[1,1].hist(mlpl2weightsNorm.flatten(), bins=hlBins, edgecolor='black', color=mlpers["cPlot"], label="Backprop")
ax[1,2].hist(mlpl3weightsNorm.flatten(), bins=hlBins, edgecolor='black', color=mlpers["cPlot"], label="Backprop")

lineStyles = [":", "--", "-"]
hexes = ["#846800", "#9c7c04", "#cba723"]
hexes_ENN = ["#5c075c", "#761f76", "#b411b4"]
for i in range(globers["nHiddenNeurons"]):
    for i2 in range(globers["nHiddenNeurons"]):
        ax[2][1].plot(splitList, ennCoefs[0, :, 0, i, i2]/np.max(np.abs(ennCoefs[0, :, 0, i, :])), 
                      color=hexes_ENN[i], lw=1.5, linestyle=lineStyles[i2])
        ax[2][2].plot(splitList, ennCoefs[0, :, 1, i, i2]/np.max(np.abs(ennCoefs[0, :, 1, i, :])), 
                      color=hexes_ENN[i], lw=1.5, linestyle=lineStyles[i2])
        ax[0][1].plot(splitList, mlpCoefs[0, :, 0, i, i2]/np.max(np.abs(mlpCoefs[0, :, 0, i, :])), 
                      color=hexes[i], lw=1.5, linestyle=lineStyles[i2])
                      #path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        ax[0][2].plot(splitList, mlpCoefs[0, :, 1, i, i2]/np.max(np.abs(mlpCoefs[0, :, 1, i, :])), 
                      color=hexes[i], lw=1.5, linestyle=lineStyles[i2])

inputBins = np.linspace(0,1,10)
#ax[0][0].hist(np.reshape(mlpInputCoefs, (-1, 512*3)).T, bins=inputBins, edgecolor='black', color=mlpGradient, alpha=0.5)
#ax[2][0].hist(np.reshape(ennInputCoefs, (-1, 512*3)).T, bins=inputBins, edgecolor='black', color=ennGradient, alpha=0.5)

import pandas as pd

mlpInputCoefs = np.reshape(mlpInputCoefs, (-1, 512*3))
ennInputCoefs = np.reshape(ennInputCoefs, (-1, 512*3))
mlpInputCoefs_PD = pd.DataFrame(mlpInputCoefs.T)
ennInputCoefs_PD = pd.DataFrame(ennInputCoefs.T)


mlpInputCoefs_PD[1].plot.kde(ax=ax[0,0], color=hexes[0], label="sparse", alpha=0.75, linestyle=lineStyles[0])
mlpInputCoefs_PD[4].plot.kde(ax=ax[0,0], color=hexes[1], label="low", alpha=0.75, linestyle=lineStyles[1])
mlpInputCoefs_PD[9].plot.kde(ax=ax[0,0], color=hexes[2], label="all",alpha=0.75, linestyle=lineStyles[2])

ennInputCoefs_PD[1].plot.kde(ax=ax[2,0], color=hexes_ENN[0], label="sparse", alpha=0.75, linestyle=lineStyles[0])
ennInputCoefs_PD[4].plot.kde(ax=ax[2,0], color=hexes_ENN[1], label="low", alpha=0.75, linestyle=lineStyles[1])
ennInputCoefs_PD[9].plot.kde(ax=ax[2,0], color=hexes_ENN[2], label="all",alpha=0.75, linestyle=lineStyles[2])

ax[0,0].set_xlim([-0.01, 0.2])
ax[2,0].set_xlim([-0.01, 0.75])
ax[1,0].set_xlim([-0.01, 0.75])
ax[3,0].set_xlim([-0.01, 0.75])

labels = ["1%", "", "100%"]
ax[0,1].set_xticks([0.01, 0.5, 1], labels=labels)
ax[0,2].set_xticks([0.01, 0.5, 1], labels=labels)
ax[0,1].set_yticks([-1, 0, 1])
ax[0,2].set_yticks([-1, 0, 1])
ax[2,1].set_xticks([0.01, 0.5, 1], labels=labels)
ax[2,2].set_xticks([0.01, 0.5, 1], labels=labels)
ax[2,1].set_yticks([-1, 0, 1])
ax[2,2].set_yticks([-1, 0, 1])

ax[0,0].set_ylabel("")
ax[1,0].set_ylabel("")
ax[2,0].set_ylabel("")

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/VariableSplitsAndReproducibility.png",
            transparent=True, 
            dpi=300)
plt.close(fig)