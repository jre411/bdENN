'''
Path: bdENS/repo/scripts/sFig1.py

Boundary-Detecting ENN Project
Accuracy Histogram
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 12-02-2024

Reference(s): 
    - bdENS/release/FigureScripts/sFig1.py

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
subprojID = "sFig1"
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

# Load the data
mlpAccs = np.load(logMan.outputDir + "/accsMLP.npy".format())
ennAccs = np.load(logMan.outputDir + "/accsENN.npy".format())

# Plot the histogram
fig, ax = plt.subplots()
nBins = 25
bins = np.linspace(0.3, 0.8, nBins)
ax.hist(mlpAccs, bins=bins, alpha=0.5, label="Backpropagation", color=cC.networkColors["mlp"], edgecolor="black")
ax.hist(ennAccs, bins=bins, alpha=0.5, label="ENN", color=cC.networkColors["enn"], edgecolor="black")
ax.set_xlabel("Test Accuracy")
ax.set_ylabel("Count")
ax.legend(bbox_to_anchor=(0.6, 0.9), loc='upper left')
ax.title.set_text("Accuracy Over Variable 85-15% Splits")

# Annotate the networks at 0.3 as unconverged
ax.annotate("Unconverged", xy=(0.33, 2050), xytext=(0.41, 2000),
             arrowprops=dict(facecolor='black', shrink=0.1, width=1, headwidth=10),
             )

# Add a line at 0.33 indicating random chance, annotate on x-axis
#ax.axvline(x=0.33, color="black", linestyle="--")
#ax.annotate("Random Chance", xy=(0.33, 0), xytext=(0.35, 3000))

# Add a line at 0.4 indicating the convergence threshold
ax.axvline(x=0.4, color="black", linestyle=":")
ax.annotate("Convergence Threshold", xy=(0.4, 0), xytext=(0.405, 2500))

# Log the average accuracy for each network and their standard deviations
convThreshold = 0.4  
ennAccs = ennAccs[mlpAccs > convThreshold]
mlpAccs = mlpAccs[mlpAccs > convThreshold]

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/sFig1.png", dpi=300)