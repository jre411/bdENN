'''
Path: bdENS/repo/scripts/sFig2.py

Boundary-Detecting ENN Project
Concept Space Retention
Author: James R. Elder
Institution: UTSW

DOO: 12-02-2024
LU: 12-02-2024

Reference(s): 
    - bdENS/release/FigureScripts/sFig2.py

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
subprojID = "sFig2"
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
mlpCM = np.load(logMan.outputDir + "/cmMLP.npy".format())
ennCM = np.load(logMan.outputDir + "/cmENN.npy".format())

# Plot confusion matrices
figWidth = 6
figHeight = 4
fig, axs = plt.subplots(1, 2, figsize=(figWidth, figHeight))

plt.subplot(1, 2, 1)
plt.imshow(ennCM, cmap="seismic", vmin=0, vmax=1)
# Annotation
for i in range(3):
    for j in range(3):
        plt.text(j, i, "{:.2f}".format(ennCM[i, j]), ha="center", va="center", color="black")
        rect = Rectangle((j-0.3, i-0.2), .6, .4, fill=True, color="white", alpha=0.5)
        plt.gca().add_patch(rect)

plt.title("ENN")
plt.xticks([0, 1, 2], ["NB", "SB", "HB"])
plt.yticks([0, 1, 2], ["NB", "SB", "HB"])
plt.ylabel("Ground Truth")
plt.xlabel("Predicted")
plt.colorbar(orientation="horizontal", label="Normalized Count")

plt.subplot(1, 2, 2)
plt.imshow(mlpCM, cmap="seismic", vmin=0, vmax=1)

# Annotation, add a white background to text
for i in range(3):
    for j in range(3):
        plt.text(j, i, "{:.2f}".format(mlpCM[i, j]), ha="center", va="center", color="black")
        rect = Rectangle((j-0.3, i-0.2), .6, .4, fill=True, color="white", alpha=0.5)
        plt.gca().add_patch(rect)
plt.xticks([0, 1, 2], ["NB", "SB", "HB"])
plt.yticks([0, 1, 2], ["NB", "SB", "HB"])

plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("Backprop")

# Add colorbar on bottom

plt.colorbar(orientation="horizontal", label="Normalized Count")
plt.tight_layout()
fig.suptitle("Confusion Matrices\n85-15% Split")

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/sFig2.png",
            transparent=True, 
            dpi=300)
plt.close(fig)