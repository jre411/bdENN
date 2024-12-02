'''
Path: bdENS/repo/scripts/sFig4.py

Boundary-Detecting ENN Project
Feature Maps
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 12-02-2024

Reference(s): 
    - bdENS/release/FigureScripts/sFig4.py

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
subprojID = "sFig4"
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
import av
import numpy as np
import matplotlib.pyplot as plt

# ENN imports
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

nFeatures = 10

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

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/{}-{}.png".format(subprojID, boundaryClass), transparent=True, dpi=300)
plt.close()

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

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/{}-{}.png".format(subprojID, boundaryClass), transparent=True, dpi=300)
plt.close()

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

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/{}-{}.png".format(subprojID, boundaryClass), transparent=True, dpi=300)
plt.close()