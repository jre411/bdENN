'''
Path: bdENS/repo/scripts/Fig5.py

Boundary-Detecting ENN Project
Fig5: Network Feature Learning
Author: James R. Elder
Institution: UTSW

DOO: 12-10-2024
LU: 12-10-2024

Reference(s): 
    - bdENS/repo/scripts/ConceptSpaceRetention.py
    - bdENS/repo/scripts/ParameterRobustness.py

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
subprojID = "Fig5"
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
    "trainTestSplit": 0.8,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3,
    "nTrials": 10, # 1000
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
    inputWeightsFrame = np.reshape(inputWeights, (4, nFeatures, 3))
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

plt.plot(np.median(accsENN, axis=0), label="ENN", color=cC.networkColors["enn"])
plt.plot(np.median(convergedMLPs, axis=0), label="Backprop", color=cC.networkColors["mlp"])
plt.fill_between(range(nFeatures), np.quantile(accsENN, 0.25, axis=0), 
                np.quantile(accsENN, 0.75, axis=0), alpha=0.2, color=cC.networkColors["enn"])
plt.fill_between(range(nFeatures), np.quantile(convergedMLPs, 0.25, axis=0), 
                np.quantile(convergedMLPs, 0.75, axis=0), alpha=0.2, color=cC.networkColors["mlp"])

plt.axhline(y=1/3, color="black", linestyle="--")
plt.text(10, 1/3+0.05, "Chance", verticalalignment="center", fontsize=8)

plt.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1, 0.1))

plt.xlabel("Features removed")
plt.xticks([0, 128, 256, 384, 512], ["0", "128", "256", "384", "512"])

plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"])
plt.ylabel("Accuracy")

# Set plot size
figWidth = 3
figHeight = 4
plt.gcf().set_size_inches(figWidth, figHeight)

plt.tight_layout()
plt.savefig(logMan.mediaDir + "/Fig5a.png",
            transparent=True, 
            dpi=300)
plt.close()

# Variables
globers = {
    "randomSeed": 42,
    "classes": ["No Boundary", "Soft Boundary", "Hard Boundary"],
    "nClasses": 3,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3,
    "nFrames": 4,
    "dimsPerFrame": 512
}

# Now for the circuit (Fig 5b)  
figWidth = 6
figHeight = 2
fig, ax = plt.subplots(1, len(globers["classes"]), figsize=(figWidth, figHeight))

connectionStates = [ [0, 0, 0], [1, 1, 0], [1, 1, 1] ]

circuitVisualizer = NV.CircuitVisualizer()
circuitVisualizer.createNeurons()
circuitVisualizer.createConnections()

for i in range(len(globers["classes"])):
    circuitVisualizer.setConnectionStates(connectionStates[i])
    circuitVisualizer.plotCircuit(ax[i])
    circuitVisualizer.annotateCircuit(ax[i])
    ax[i].set_title(globers["classes"][i])

plt.tight_layout()
fig.savefig(logMan.mediaDir + "/Fig5b.png",
            transparent=True, 
            dpi=300)
plt.close(fig)

globers = {
    "randomSeed": 42,
    "classes": ["No Boundary", "Soft Boundary", "Hard Boundary"],
    "classAbbrevs": ["NB", "SB", "HB"],
}

xAll, yAll = cF.loadEmbeddings()
mdlENN = lB.main_0(xAll, yAll, 0, 3)

inputWeights = mdlENN.layers[0].weights
inputWeightsFrame = np.reshape(inputWeights, (4, 512, 3))
inputWeightsFrameAvg = np.mean(np.abs(inputWeightsFrame), axis=0)
inputWeightsFrameAvgAvg = np.mean(inputWeightsFrameAvg, axis=1)
sortedIndices = np.argsort(-inputWeightsFrameAvgAvg)

# Visualize the feature map for the first frame
import matplotlib.pyplot as plt
import av

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# Load an image 
boundaryClass = "NB"
index = 1
nFrames = 4

image = av.open(logMan.inputDir + "/videoFrames/{}/{}{}-{}.png".format(boundaryClass, boundaryClass, index, 0))
img = image.decode(video=0)
for frame in img:
    img = frame.to_ndarray()

# Load the VGG16 model
vgg16 = VGG16(input_shape=img.shape, include_top=False)

nFeatures = 5

figWidth = 1.5*nFeatures
figHeight = 1*nFrames
fig, ax = plt.subplots(nFrames, nFeatures+1, figsize=(figWidth, figHeight))
ax[0,0].set_title("{}".format(boundaryClass))

# First column is original 4 frames, second column is feature map
for i in range(nFrames):
    img = av.open(logMan.inputDir + "/videoFrames/{}/{}{}-{}.png".format(boundaryClass, boundaryClass, index, i)).decode(video=0)
    for frame in img:
        img = frame.to_ndarray()
    imgPP = preprocess_input(img)
    featureMap = vgg16.predict(np.array([imgPP]))
    for j in range(nFeatures):
        ax[i, 0].imshow(img)
        ax[i, 1+j].imshow(featureMap[0, :, :, sortedIndices[j]], cmap="gray")
        ax[i, 0].axis("off")
        ax[i, 1+j].axis("off")
        ax[0,j+1].set_title("{}".format(sortedIndices[j]))

plt.tight_layout()
fig.savefig(logMan.mediaDir + "/Fig5c{}.png".format(boundaryClass),
            transparent=True, 
            dpi=300)
plt.close(fig)

# SB
boundaryClass = "SB"
index = 1

figWidth = 1.5*nFeatures
figHeight = 1*nFrames
fig, ax = plt.subplots(nFrames, nFeatures+1, figsize=(figWidth, figHeight))
ax[0,0].set_title("{}".format(boundaryClass))

# First column is original 4 frames, second column is feature map
for i in range(nFrames):
    img = av.open(logMan.inputDir + "/videoFrames/{}/{}{}-{}.png".format(boundaryClass, boundaryClass, index, i)).decode(video=0)
    for frame in img:
        img = frame.to_ndarray()
    imgPP = preprocess_input(img)
    featureMap = vgg16.predict(np.array([imgPP]))
    for j in range(nFeatures):
        ax[i, 0].imshow(img)
        ax[i, 1+j].imshow(featureMap[0, :, :, sortedIndices[j]], cmap="gray")
        ax[i, 0].axis("off")
        ax[i, 1+j].axis("off")
        ax[0,j+1].set_title("{}".format(sortedIndices[j]))

plt.tight_layout()
fig.savefig(logMan.mediaDir + "/Fig5c{}.png".format(boundaryClass),
            transparent=True, 
            dpi=300)
plt.close(fig)

# HB
boundaryClass = "HB"
index = 1

figWidth = 1.5*nFeatures
figHeight = 1*nFrames
fig, ax = plt.subplots(nFrames, nFeatures+1, figsize=(figWidth, figHeight))

ax[0,0].set_title("{}".format(boundaryClass))
# First column is original 4 frames, second column is feature map
for i in range(nFrames):
    img = av.open(logMan.inputDir + "/videoFrames/{}/{}{}-{}.png".format(boundaryClass, boundaryClass, index, i)).decode(video=0)
    for frame in img:
        img = frame.to_ndarray()
    imgPP = preprocess_input(img)
    featureMap = vgg16.predict(np.array([imgPP]))
    for j in range(nFeatures):
        ax[i, 0].imshow(img)
        ax[i, 1+j].imshow(featureMap[0, :, :, sortedIndices[j]], cmap="gray")
        ax[i, 0].axis("off")
        ax[i, 1+j].axis("off")
        ax[0,j+1].set_title("{}".format(sortedIndices[j]))

plt.tight_layout()
fig.savefig(logMan.mediaDir + "/Fig5c{}.png".format(boundaryClass),
            transparent=True, 
            dpi=300)
plt.close(fig)