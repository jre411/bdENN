'''
Path: bdENS/repo/scripts/NetworkVisualization.py

Boundary-Detecting ENN Project
Network Visualization
- Figure 1e: Top Model Visualization
Author: James R. Elder
Institution: UTSW

DOO: 11-07-2024
LU: 11-07-2024

Reference: bdENS/develop/FigureScripts/Fig1e.py

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
subprojID = "NetworkVisualization"
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
import matplotlib.pyplot as plt

# ENN imports

# Local imports
from utils import LumberJack as LJ
from utils import NetworkVisualizer as NV
from utils import commonFunctions as cF

# Logging
logMan = LJ.LoggingManager(subprojID)
cF.updatePlotrcs(font="serif")

# Variables
globers = {
    "randomSeed": 42,
    "nClasses": 3,
    "nHiddenLayers": 2,
    "nHiddenNeurons": 3,
    "nFrames": 4,
    "dimsPerFrame": 512
}

# Plot a representative network (Fig 1e)
figWidth = 6
figHeight = 3
fig, ax = plt.subplots(1, 1, figsize=(figWidth, figHeight))

# Initialize the network visualizer
networkVisualizer = NV.NetworkVisualizer(inputDims=globers["dimsPerFrame"], 
                               nFrames=globers["nFrames"],
                               nHiddenLayers=globers["nHiddenLayers"], 
                               nHiddenNeurons=globers["nHiddenNeurons"],
                               outputDims=globers["nClasses"], 
                               )

# Set plot parameters
networkVisualizer.nInputRows = 32

# Create the neurons and connections
networkVisualizer.createNeurons()
networkVisualizer.createConnections()

# Plot the network
networkVisualizer.plotNetwork(ax=ax)
networkVisualizer.annotateNetwork(ax=ax)

# Figure settings
ax.set_xlim(-1, networkVisualizer.nInputRows * 2 + 1)
ax.set_ylim(-1, networkVisualizer.nInputRows + 1)
ax.axis("off")
fig.tight_layout()

# Save the figure
fig.savefig(logMan.mediaDir + "/Fig1e.png",
            transparent=True, 
            dpi=300)
plt.close(fig)