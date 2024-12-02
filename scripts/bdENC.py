'''
Path: bdENS/repo/scripts/bdENC.py

Boundary-Detecting ENN Project
bdENC Performance Evaluation
Author: James R. Elder
Institution: UTSW

DOO: 11-29-2024
LU: 11-29-2024

Reference(s): 
    - bdENS/develop/scripts/xplor/bdENC.py

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
subprojID = "bdENC"
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
    "trainTestSplit": 1,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3,
}

enners = {
    "subconceptSelector": 0,
}

xAll, yAll = cF.loadEmbeddings()
nFeatures = 512

# Minimal feature set
topFeatures = [7, 9, 10, 12, 13, 17, 19, 21, 22, 23, 26, 28, 29, 30, 56, 99, 119, 155, 191, 
               210, 216, 225, 232, 237, 254, 269, 285, 302, 315, 424, 428, 430, 480, 482, 483, 
               485, 486, 487, 488, 489, 490, 493, 494, 495, 499, 501, 502, 503, 507, 511]

# Create a sparse dataset
xAllSparse = xAll.copy()
for i in range(nFeatures):
    if (i not in topFeatures):
        featureFocus = [i, i+nFeatures, i+2*nFeatures, i+3*nFeatures]
        xAllSparse[:, featureFocus] = np.zeros((xAllSparse.shape[0], 4))
xAllSparse = xAllSparse[:, np.sum(xAllSparse, axis=0) != 0]

# Minimal circuit class
class minimalCircuit:
    def __init__(self, _weights, _thresholds):
        self.weights = _weights
        self.weights[:,2] = -self.weights[:,2]
        self.thresholds = _thresholds
        self.thresholds[:2] = -self.thresholds[:2]

    def predict(self, _x):
        _output = np.dot(_x, self.weights)
        if not (int(_output[0] > self.thresholds[0]) and int(_output[1] > self.thresholds[1])):
            return int(_output[2] > self.thresholds[2]) + 1
        else:
            return 0
        
    def evaluate(self, _x, _y):
        _counter = 0
        for i in range(_y.shape[0]):
            _predictedLabel = self.predict(_x[i])
            if _predictedLabel != _y[i]:
                #print("Predicted: {}, Actual: {}".format(_predictedLabel, _y[i]))
                _counter += 1
        return 1 - _counter/_y.shape[0]

from sklearn.linear_model import LogisticRegression

mdlNames = ["ENN", "ENC", "MLP", "LogReg"]
minSplit = 0.05
maxSplit = 0.9
nSplits = 10
splits = np.linspace(minSplit, maxSplit, nSplits)

nTrials = 100

mdlAccs = np.zeros((len(mdlNames), nSplits, nTrials))

for i2 in range(nTrials):

    for i in range(nSplits):
        xTrain, _, yTrain, _ = train_test_split(xAllSparse, yAll, test_size=1-splits[i], random_state=globers["randomSeed"]+i2)
        
        # ENN
        mdlENN = lB.main_0(xTrain, yTrain, enners["subconceptSelector"], globers["nHiddenNeurons"])
        acc = 1 - mdlENN.compute_error(xAllSparse, yAll)[0]
        mdlAccs[0, i, i2] = acc
        
        # ENC
        bdENC = minimalCircuit(mdlENN.layers[0].weights, mdlENN.layers[0].biases)
        accuracy = bdENC.evaluate(xAllSparse, yAll)
        mdlAccs[1, i, i2] = accuracy
        
        # MLP
        mdlMLP = MLPClassifier(hidden_layer_sizes=(globers["nHiddenNeurons"],)*globers["nHiddenLayers"], max_iter=10000, random_state=globers["randomSeed"]+i+2*i2)
        mdlMLP.fit(xTrain, yTrain)
        acc = mdlMLP.score(xAllSparse, yAll)
        mdlAccs[2, i, i2] = acc
        
        # LogReg
        mdlLogReg = LogisticRegression(random_state=globers["randomSeed"])
        mdlLogReg.fit(xTrain, yTrain)
        acc = mdlLogReg.score(xAllSparse, yAll)
        mdlAccs[3, i, i2] = acc


print("Model accuracies:")  
print(mdlAccs)

# Plot accuracies
figWidth = 6
figHeight = 4
fig, ax = plt.subplots(figsize=(figWidth, figHeight))

for i in range(len(mdlNames)):
    meanAcc = np.mean(mdlAccs[i], axis=1)
    stdAcc = np.std(mdlAccs[i], axis=1)
    ax.plot(splits, meanAcc, label=mdlNames[i])
    ax.fill_between(splits, meanAcc-stdAcc, meanAcc+stdAcc, alpha=0.2)

ax.set_xlabel("# Training Samples")
ax.set_xticks(splits, labels=[str(int(splits[i]*xAllSparse.shape[0])) for i in range(nSplits)])
ax.set_ylabel("Entire Dataset Accuracy")
ax.set_title("Model Performance on Top 50 ENN Features")
ax.legend()

fig.subplots_adjust(wspace=0.75)
fig.savefig(logMan.mediaDir + "/sFig5.png",
            transparent=True, 
            dpi=300)
plt.close(fig)