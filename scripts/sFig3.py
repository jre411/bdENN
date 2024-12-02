'''
Path: bdENS/repo/scripts/sFig3.py

Boundary-Detecting ENN Project
Class performace under noise
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 12-02-2024

Reference(s): 
    - bdENS/release/FigureScripts/sFig3.py

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
subprojID = "sFig3"
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
from matplotlib.patches import Rectangle

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

# Load the data
mlpAccs = np.load(logMan.outputDir + "/mlpClassAccs_1n1000s10000t-42r.npy".format())
ennAccs = np.load(logMan.outputDir + "/ennClassAccs_1n1000s10000t-42r.npy".format())


# Plot confusion matrices
figWidth = 6
figHeight = 3
fig, ax = plt.subplots(1, 2, figsize=(figWidth, figHeight), sharex=True)

xticks = range(0, 1250, 250)
xticklabels = [str(i) for i in np.arange(0, 1.25, 0.25)]

# ENN
ax[0].plot(np.median(ennAccs[:,:,0], axis=0), color=cC.conceptColors[0], label="No Boundary")
ax[0].plot(np.median(ennAccs[:,:,1], axis=0), color=cC.conceptColors[1], label="Soft Boundary")
ax[0].plot(np.median(ennAccs[:,:,2], axis=0), color=cC.conceptColors[2], label="Hard Boundary")
ax[0].set_xticks(xticks, xticklabels)
ax[0].set_yticks(np.arange(0.7, 1.0, 0.05))
ax[0].title.set_text("ENN")
ax[0].set_ylabel("Retention Accuracy")
ax[0].set_xlabel("Fractional Noise")
ax[0].legend()

# MLP
ax[1].plot(np.median(mlpAccs[:,:,0], axis=0), color=cC.conceptColors[0], label="No Boundary")
ax[1].plot(np.median(mlpAccs[:,:,1], axis=0), color=cC.conceptColors[1], label="Soft Boundary")
ax[1].plot(np.median(mlpAccs[:,:,2], axis=0), color=cC.conceptColors[2], label="Hard Boundary")
ax[1].title.set_text("Backprop")
ax[1].set_xticks(xticks, xticklabels)
ax[1].set_yticks(np.arange(0.7, 1.0, 0.05))
ax[1].set_ylabel("Retention Accuracy")
ax[1].set_xlabel("Fractional Noise")
ax[1].legend()

fig.suptitle("Concept Retention")

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/sFig3.png",
            transparent=True, 
            dpi=300)
plt.close(fig)