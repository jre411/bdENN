'''
Path: bdENS/repo/scripts/ConceptSpaceFormation.py

Boundary-Detecting ENN Project
Concept Space Formation
Author: James R. Elder
Institution: UTSW

DOO: 02-07-2025
LU: 02-07-2025

Reference(s): 
    - bdENS/repo/scripts/ConceptSpaceRetention.py

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
subprojID = "ConceptSpaceFormation"
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

'''
figWidth = 15
figHeight = 6
fig, ax = plt.subplots(2, 5, figsize=(figWidth, figHeight), subplot_kw={'projection': '3d'})

# Plot the decision boundaries
nSteps = 5
samplers = np.linspace(-1, 1, nSteps)
samplerAlpha = 0.05

splits = [0.01,  0.1, 0.25, 0.5, 0.8]

for i in range(len(splits)):
    xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, 
                                                    train_size=splits[i],
                                                    random_state=globers["randomSeed"], 
                                                    stratify=yAll)
    print("Split: {}".format(splits[i]))
    print("Train size: {}".format(xTrain.shape[0]))

    mdlENN = lB.main_0(xTrain, yTrain, enners["subconeptSelector"], globers["nHiddenNeurons"])

    # Pick 100 random test inputs
    testers = np.random.choice(range(xTest.shape[0]), 100)
    for j in testers:
        l1Output = mdlENN.layers[0].compute_output(xTest[j])
        l2Output = mdlENN.layers[1].compute_output(l1Output)[0]
        y = np.argmax(mdlENN.compute_output(xTest[j]))  
        if y == yTest[j]:
            _marker = "o"
        else:
            _marker = "x"
        ax[0,i].scatter(l1Output[0][0], l1Output[0][1], l1Output[0][2], 
                      color=cC.conceptColors[yTest[j]], marker=_marker,
                      alpha=0.75)
        ax[1,i].scatter(l2Output[0], l2Output[1], l2Output[2], 
                      color=cC.conceptColors[yTest[j]], marker=_marker,
                      alpha=0.75)
    
    # Add samplers to visualize the decision boundaries
    for j in samplers:
        for k in samplers:
            for l in samplers:
                l1Output = mdlENN.layers[1].compute_output(np.array([j, k, l]))
                y = np.argmax(mdlENN.layers[-1].compute_output(l1Output))
                ax[0,i].scatter(j, k, l, color=cC.conceptColors[y], alpha=samplerAlpha)
                y = np.argmax(mdlENN.layers[-1].compute_output(np.array([j, k, l])))  
                ax[1,i].scatter(j, k, l, color=cC.conceptColors[y], alpha=samplerAlpha)

# Annotate the plots
for i in range(len(splits)):
    for i2 in range(2):
        # Set limits
        ax[i2,i].set_xticks([-1, 0, 1])
        ax[i2,i].set_yticks([-1, 0, 1])
        ax[i2,i].set_zticks([-1, 0, 1])
        
        # Set title
        ax[0,i].set_title("{}".format(splits[i]))
        
        # Set view
        ax[1,i].view_init(elev=20, azim=45)
        ax[0,i].view_init(elev=20, azim=-45)

# Save the figure
fig.tight_layout()
fig.savefig(logMan.mediaDir + "/ConceptSpaceFormationENN.png",
            transparent=True, 
            dpi=300)
plt.close(fig)

'''

mdlMLP = MLPClassifier(hidden_layer_sizes=(globers["nHiddenNeurons"], globers["nHiddenNeurons"]), 
                       max_iter=mlpers["maxIter"], 
                       random_state=9, 
                       activation=mlpers["activation"])


figWidth = 15
figHeight = 6
fig, ax = plt.subplots(2, 5, figsize=(figWidth, figHeight), subplot_kw={'projection': '3d'})

# Plot the decision boundaries
nSteps = 20  
samplers = np.linspace(-1, 1, nSteps)
samplerAlpha = 0.05

splits = [0.01,  0.1, 0.25, 0.5, 0.8]

for i in range(len(splits)):
    xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, 
                                                    train_size=splits[i],
                                                    random_state=globers["randomSeed"], 
                                                    stratify=yAll)
    print("Split: {}".format(splits[i]))
    print("Train size: {}".format(xTrain.shape[0]))

    mdlMLP.fit(xTrain, yTrain)

    # Pick 100 random test inputs
    testers = np.random.choice(range(xTest.shape[0]), 100)
    for j in testers:
        l1Output = mdlMLP.coefs_[0].T @ xTest[j] + mdlMLP.intercepts_[0]
        l1Output = np.tanh(l1Output)
        l2Output = mdlMLP.coefs_[1].T @ l1Output + mdlMLP.intercepts_[1]
        l2Output = np.tanh(l2Output)

        y = np.argmax(mdlMLP.predict_proba(xTest[j].reshape(1, -1))[0])
        if y == yTest[j]:
            _marker = "o"
        else:
            _marker = "x"
        ax[0,i].scatter(l1Output[0], l1Output[1], l1Output[2], 
                      color=cC.conceptColors[yTest[j]], marker=_marker,
                      alpha=0.75)
        ax[1,i].scatter(l2Output[0], l2Output[1], l2Output[2], 
                      color=cC.conceptColors[yTest[j]], marker=_marker,
                      alpha=0.75)
    
    # Add samplers to visualize the decision boundaries
    for j in samplers:
        for k in samplers:
            for l in samplers:
                l1Output = mdlMLP.coefs_[1].T @ np.array([j, k, l]) + mdlMLP.intercepts_[1]
                l1Output = np.tanh(l1Output)
                y = np.argmax(mdlMLP.coefs_[2].T @ l1Output + mdlMLP.intercepts_[2])
                ax[0,i].scatter(j, k, l, color=cC.conceptColors[y], alpha=samplerAlpha)
                y = np.argmax(mdlMLP.coefs_[2].T @ np.array([j, k, l]) + mdlMLP.intercepts_[2])
                ax[1,i].scatter(j, k, l, color=cC.conceptColors[y], alpha=samplerAlpha)

# Annotate the plots
for i in range(len(splits)):
    for i2 in range(2):
        # Set limits
        ax[i2,i].set_xticks([-1, 0, 1])
        ax[i2,i].set_yticks([-1, 0, 1])
        ax[i2,i].set_zticks([-1, 0, 1])
        
        # Set title
        ax[0,i].set_title("{}".format(splits[i]))
        
        # Set view
        ax[1,i].view_init(elev=20, azim=45)
        ax[0,i].view_init(elev=20, azim=-45)

# Save the figure
fig.tight_layout()
fig.savefig(logMan.mediaDir + "/ConceptSpaceFormationMLP.png",
            transparent=True, 
            dpi=300)
plt.close(fig)
