'''
Path: bdENS/repo/scripts/VariableTrainingData.py

Boundary-Detecting ENN Project
Variable Training Data
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 12-02-2024

Reference(s): 
    - bdENS/develop/jobs/2024/July/scripts/archive/May/fig2b.py

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

# Logging
logMan = LJ.LoggingManager(subprojID)
cF.updatePlotrcs(font="serif")

nSplits = 10000
randomSeed = 42

# Dataset (VGG-embedded) and labels
xAll, yAll = cF.loadEmbeddings()
print("Loaded {} inputs and {} labels".format(len(xAll), len(yAll)))
print("Input dimensionality: {}".format(xAll.shape[1]))
print("Labels are: {}".format(np.unique(yAll)))

inputDims = xAll.shape[1]
hiddenLayerSize = 3

l1weightsENN = np.zeros((nSplits, inputDims, hiddenLayerSize))
l2weightsENN = np.zeros((nSplits, hiddenLayerSize, hiddenLayerSize))
l3weightsENN = np.zeros((nSplits, hiddenLayerSize, len(np.unique(yAll))))

l1weightsMLP = np.zeros((nSplits, inputDims, hiddenLayerSize))
l2weightsMLP = np.zeros((nSplits, hiddenLayerSize, hiddenLayerSize))
l3weightsMLP = np.zeros((nSplits, hiddenLayerSize, len(np.unique(yAll))))

mdlMLP = MLPClassifier(hidden_layer_sizes=(hiddenLayerSize,hiddenLayerSize), max_iter=1000, random_state=randomSeed)
accsMLP = np.zeros(nSplits)
accsENN = np.zeros(nSplits)

for i in range(nSplits):
    xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, test_size=0.15, random_state=randomSeed+i, stratify=yAll)
    
    mdlMLP.fit(xTrain, yTrain)
    mdlENN = lB.main_0(xTrain, yTrain, 0, hiddenLayerSize)

    l1weightsENN[i] = mdlENN.layers[0].weights
    l2weightsENN[i] = mdlENN.layers[1].weights
    l3weightsENN[i] = mdlENN.layers[2].weights

    l1weightsMLP[i] = mdlMLP.coefs_[0]
    l2weightsMLP[i] = mdlMLP.coefs_[1]
    l3weightsMLP[i] = mdlMLP.coefs_[2]


    accsMLP[i] = mdlMLP.score(xTest, yTest)
    accsENN[i] = 1 - mdlENN.compute_error(xTest, yTest)[0]

np.save(logMan.outputDir + "/accsMLP.npy", accsMLP)
np.save(logMan.outputDir + "/accsENN.npy", accsENN)

np.save(logMan.outputDir + "/l1weightsENN.npy", l1weightsENN)
np.save(logMan.outputDir + "/l2weightsENN.npy", l2weightsENN)
np.save(logMan.outputDir + "/l3weightsENN.npy", l3weightsENN)

np.save(logMan.outputDir + "/l1weightsMLP.npy", l1weightsMLP)
np.save(logMan.outputDir + "/l2weightsMLP.npy", l2weightsMLP)
np.save(logMan.outputDir + "/l3weightsMLP.npy", l3weightsMLP)