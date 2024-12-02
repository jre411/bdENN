'''
Path: bdENS/repo/scripts/NoiseInjection.py

Boundary-Detecting ENN Project
Noise Injection
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 12-02-2024

Reference(s): 
    - bdENS/develop/jobs/NoiseInjection/2024-08-14/NoiseInjection.py

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
projDir = "/project/{}/{}/{}/{}/".format(labAffiliation, labID, user, projID)

# Subproject variables
subprojID = "NoiseInjection"
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

# Logging
logMan = LJ.LoggingManager(subprojID)

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

enners = {
    "subconceptSelector": 0,
    
}

mlpers = {
    "maxIter": 10000,
    "activation": "tanh",
}

saveSuffix = "{}n{}s{}t-{}r".format(globers["maxNoise"], globers["nSteps"], globers["nTrials"], globers["randomSeed"])

logMan.logParams(globers, "Global variables")
logMan.logParams(enners, "ENN variables")
logMan.logParams(mlpers, "MLP variables")

xAll, yAll = cF.loadEmbeddings()

ennAccs = np.zeros((globers["nTrials"], globers["nSteps"]))
ennClassAccs = np.zeros((globers["nTrials"], globers["nSteps"], globers["nClasses"]))
mlpAccs = np.zeros((globers["nTrials"], globers["nSteps"]))
mlpClassAccs = np.zeros((globers["nTrials"], globers["nSteps"], globers["nClasses"]))

xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, train_size=globers["trainTestSplit"], random_state=globers["randomSeed"])

# Train model(s)
mdlENN = lB.main_0(xTrain, yTrain, 0, globers["nHiddenNeurons"])
mdlMLP = MLPClassifier(hidden_layer_sizes=(globers["nHiddenNeurons"], globers["nHiddenNeurons"]), max_iter=mlpers["maxIter"], random_state=globers["randomSeed"], activation="tanh")
mdlMLP.fit(xTrain, yTrain)

# Get TP
ennTPs = cF.getTruePosInds(mdlENN, xTest, yTest, mdlType="enn")
mlpTPs = cF.getTruePosInds(mdlMLP, xTest, yTest, mdlType="mlp")

# All xTest TP where yTest == 0, 1, 2
ennTP_NB = []
ennTP_SB = []
ennTP_HB = []

mlpTP_NB = []
mlpTP_SB = []
mlpTP_HB = []

for j in range(len(ennTPs)):
    if yTest[ennTPs[j]] == 0:
        ennTP_NB.append(ennTPs[j])
    elif yTest[ennTPs[j]] == 1:
        ennTP_SB.append(ennTPs[j])
    elif yTest[ennTPs[j]] == 2:
        ennTP_HB.append(ennTPs[j])

for j in range(len(mlpTPs)):
    if yTest[mlpTPs[j]] == 0:
        mlpTP_NB.append(mlpTPs[j])
    elif yTest[mlpTPs[j]] == 1:
        mlpTP_SB.append(mlpTPs[j])
    elif yTest[mlpTPs[j]] == 2:
        mlpTP_HB.append(mlpTPs[j])

# Main loop
for i in range(globers["nTrials"]):

    # Noise injection loop
    for j in range(globers["nSteps"]):
        noise = j * globers["maxNoise"] / globers["nSteps"]
        
        # Copy OG models
        noisyENN = mdlENN.copy()
        noisyMLP = deepcopy(mdlMLP) 

        # Inject noise
        for _layer in globers["layerList"]:
            noisyENN.layers[_layer].weights = mdlENN.layers[_layer].weights + mdlENN.layers[_layer].weights * noise * np.random.normal(size=mdlENN.layers[_layer].weights.shape)
            noisyMLP.coefs_[_layer] = mdlMLP.coefs_[_layer] + mdlMLP.coefs_[_layer] * noise * np.random.normal(size=mdlMLP.coefs_[_layer].shape)

        # Evaluate model on TP
        ennAccs[i, j] = 1 - noisyENN.compute_error(xTest[ennTPs], yTest[ennTPs])[0]
        mlpAccs[i, j] = noisyMLP.score(xTest[mlpTPs], yTest[mlpTPs])

        # Evaluate model on classes
        ennClassAccs[i, j, 0] = 1 - noisyENN.compute_error(xTest[ennTP_NB], yTest[ennTP_NB])[0]
        ennClassAccs[i, j, 1] = 1 - noisyENN.compute_error(xTest[ennTP_SB], yTest[ennTP_SB])[0]
        ennClassAccs[i, j, 2] = 1 - noisyENN.compute_error(xTest[ennTP_HB], yTest[ennTP_HB])[0]

        mlpClassAccs[i, j, 0] = noisyMLP.score(xTest[mlpTP_NB], yTest[mlpTP_NB])
        mlpClassAccs[i, j, 1] = noisyMLP.score(xTest[mlpTP_SB], yTest[mlpTP_SB])
        mlpClassAccs[i, j, 2] = noisyMLP.score(xTest[mlpTP_HB], yTest[mlpTP_HB])

    # Save intermediate results
    np.save(logMan.outputDir + "/ennAccs_{}.npy".format(saveSuffix), ennAccs)
    np.save(logMan.outputDir + "/mlpAccs_{}.npy".format(saveSuffix), mlpAccs)

# Save final results
np.save(logMan.outputDir + "/ennAccs_{}.npy".format(saveSuffix), ennAccs)
np.save(logMan.outputDir + "/mlpAccs_{}.npy".format(saveSuffix), mlpAccs)

# Save class accuracies
np.save(logMan.outputDir + "/ennClassAccs_{}.npy".format(saveSuffix), ennClassAccs)
np.save(logMan.outputDir + "/mlpClassAccs_{}.npy".format(saveSuffix), mlpClassAccs)
