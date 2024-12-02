'''
Path: bdENS/repo/scripts/PlotMaker.py

Boundary-Detecting ENN Project
Results Plots 
- Figure 2a: Boundary Cell Firing
- Figure 2b: Event Cell Firing
- Figure 2c: Average Firing Rates
- Figure 2d: Trial Firings
- Figure 2e: Neuron Average Firing
- Figure 3b: MLP Plots
- Figure 3c: ENN Plots
- Figure 3e: Network Robustness Plots
- Figure 3f: Concept Space Plots
Author: James R. Elder
Institution: UTSW

DOO: 11-25-2024
LU: 11-25-2024

Reference(s): 
    - bdENS/release/FigureScripts/Fig2abc.py
    - bdENS/release/FigureScripts/Fig2d.py
    - bdENS/release/FigureScripts/Fig2e.py
    - bdENS/release/FigureScripts/Fig3b.py
    - bdENS/release/FigureScripts/Fig3c.py
    - bdENS/release/FigureScripts/Fig3e.py
    - bdENS/release/FigureScripts/Fig3f.py

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
import numpy as np
import matplotlib.pyplot as plt

# ENN imports

# Local imports
from utils import LumberJack as LJ
from utils import NetworkVisualizer as NV
from utils import commonFunctions as cF
from utils import commonClasses as cC

# Logging
logMan = LJ.LoggingManager(subprojID)
cF.updatePlotrcs(font="serif")

# Variables
globers = {
    "randomSeed": 42,
    "nTrials": 135,
    "nBins": 150,
    "msPerBin": 10,
    "boundaryBin": 50,
}

eventers = {
    "nNeurons": 36,
    "cellType": "HB",
    "peakFiring": 30.1,
    "peakStd": 5.5,
}

trialers = {
    "nNBTrials": 30,
    "nSBTrials": 75,
    "nHBTrials": 30
}

boundaryers = {
    "nNeurons": 42,
    "cellType": "B",
    "peakFiring": 19.7,
    "peakStd": 4.9
}

boundaryRecording = cF.loadNeuron(7, boundaryers["cellType"])
eventRecording = cF.loadNeuron(2, eventers["cellType"])

eventCellMeanRecordings = np.zeros((eventers["nNeurons"], globers["nBins"]))
boundaryCellMeanRecordings = np.zeros((boundaryers["nNeurons"], globers["nBins"]))

for i in range(1, eventers["nNeurons"]+1):
    neuronRecording = cF.loadNeuron(i, 'HB')
    neuronRecording = np.mean(neuronRecording[105:], axis=0)
    eventCellMeanRecordings[i-1] = neuronRecording

eventCellsMeanRecordings = np.mean(eventCellMeanRecordings, axis=0)

for i in range(1, boundaryers["nNeurons"]+1):
    neuronRecording = cF.loadNeuron(i, 'B')
    neuronRecording = np.mean(neuronRecording[30:], axis=0)
    boundaryCellMeanRecordings[i-1] = neuronRecording

boundaryCellsMeanRecordings = np.mean(boundaryCellMeanRecordings, axis=0)

figWidth = 8
figHeight = 2.5
fig, ax = plt.subplots(1,3, figsize=(figWidth, figHeight))

randomSBTrials = np.random.choice(np.arange(trialers["nNBTrials"], trialers["nNBTrials"] + trialers["nSBTrials"]), 30, replace=False)
boundaryRecordingSubSample = np.append(boundaryRecording[:30], boundaryRecording[randomSBTrials])
boundaryRecordingSubSample = np.append(boundaryRecordingSubSample, boundaryRecording[105:])
boundaryRecordingSubSample = np.reshape(boundaryRecordingSubSample, (-1, 150))
ax[0].imshow(boundaryRecordingSubSample, aspect='auto', cmap="gray_r")
ax[0].set_title("Boundary Cell (B)")

randomSBTrials = np.random.choice(np.arange(trialers["nNBTrials"], trialers["nNBTrials"] + trialers["nSBTrials"]), 30, replace=False)
eventRecordingSubSample = np.append(eventRecording[:30], eventRecording[randomSBTrials])
eventRecordingSubSample = np.append(eventRecordingSubSample, eventRecording[105:])
eventRecordingSubSample = np.reshape(eventRecordingSubSample, (-1, 150))
ax[1].imshow(eventRecordingSubSample, aspect='auto', cmap="gray_r")
ax[1].set_title("Event Cell (E)")

ax[2].plot(boundaryCellsMeanRecordings, color=cC.cellColors["Boundary"], label="B", linestyle="-", linewidth=2.5)
ax[2].plot(eventCellsMeanRecordings, color=cC.cellColors["Event"], label="E", linestyle="-", linewidth=2.5)
ax[2].legend(loc="upper right", ncol=1, bbox_to_anchor=(1.1, 1.05))
# Add early at the top of the plot on the left, and late at the top of the plot on the right
ax[2].annotate("Early", xy=(globers["boundaryBin"], 0.27), xytext=(globers["boundaryBin"]+10, 0.27), textcoords="data", ha="center", va="center", fontsize=8, rotation=90)
ax[2].annotate("Late", xy=(globers["boundaryBin"]+22, 0.275), xytext=(globers["boundaryBin"]+32, 0.275), textcoords="data", ha="center", va="center", fontsize=8, rotation=90) 
ax[2].set_title("Mean Firing Rate")

for i in range(3):
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)    
    ax[i].set_xticks([0, 50, 100, 150], ["-0.5", "0.0", "0.5", "1.0"])
    ax[i].set_xlim(0, globers["nBins"])
    ax[i].set_xlabel("Time, relative to\nboundary (s)")
    ax[i].axvline(globers["boundaryBin"], color="black", linestyle="-")
    if i != 2:
        ax[i].set_ylabel("Trial")
        ax[i].set_ylim(90, 0)
        ax[i].set_yticks([0, 30, 60, 90], [])
        ax[i].axhspan(0, trialers["nNBTrials"], color=cC.conceptColors[0], alpha=0.1)
        ax[i].axhspan(trialers["nNBTrials"], trialers["nNBTrials"] + 30, color=cC.conceptColors[1], alpha=0.1)
        ax[i].axhspan(60, 90, color=cC.conceptColors[2], alpha=0.1)
    else:
        ax[i].set_yticks([0, 0.1, 0.2, 0.3], ["0", "1", "2", "3"])
        ax[i].set_ylabel("Firing Rate (Hz)")
        ax[i].axvline(globers["boundaryBin"]+22, color="black", linestyle="--")

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/Fig2abc.png",
            transparent=True, 
            dpi=300)
plt.close(fig)

# Figure 2d: Trial Firings

# ENN imports
from enn.network import Network
import enn.learnBoundaries as lB

# 3rd party imports
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

"""
# Variables
globers = {
    "randomSeed": 42,
    "nTrials": 135,
    "nBins": 150,
    "msPerBin": 10,
    "boundaryBin": 50,
    "trainTestSplit": 0.5,
    "nHiddenNeurons": 3,
    "nHiddenLayers": 2
}

mlpers = {
    "maxIter": 10000,
    "activation": "tanh",
}

eventers = {
    "nNeurons": 36,
    "cellType": "HB",
    "peakFiring": 30.1,
    "peakStd": 5.5,
}

trialers = {
    "nNBTrials": 30,
    "nSBTrials": 75,
    "nHBTrials": 30
}

enners = {
    "subconceptSelector": 0,
}

boundaryers = {
    "nNeurons": 42,
    "cellType": "B",
    "peakFiring": 19.7,
    "peakStd": 4.9
}

plotters = {
    "aTrial": 0.1,
    "cBoundary": "#E8BDCB",
    "cEvent": "#E5C8EE"
}

# Functions
def activateFiring(_avgPeakFiring, _backgroundFiring):
    _scaleFactor = 0.5
    _xOffset = 1.5
    _steepness = 0.1 / np.std(np.mean(_backgroundFiring, axis=1))
    _yOffset = 0.5

    _normalizedFiring = _avgPeakFiring / np.mean(_backgroundFiring)

    return _scaleFactor * np.tanh(_steepness * (_normalizedFiring - _xOffset)) + _yOffset


eventCellActivatedFirings = np.zeros((globers["nTrials"], eventers["nNeurons"]))
lateBoundaryInds = []
earlyEventInds = []

cutoff = 22
for i in range(eventers["nNeurons"]):
    recording = cF.loadNeuron(i+1, eventers["cellType"])
    preBoundaryFiring = recording[:,:globers["boundaryBin"]]
    peakFiring = recording[:,int(globers["boundaryBin"]+eventers["peakFiring"]-2*eventers["peakStd"]):int(globers["boundaryBin"] + eventers["peakFiring"]+2*eventers["peakStd"])]  
    avgPeakFiring = np.mean(peakFiring, axis=1)
    if np.argmax(np.mean(recording[105:], axis=0)) <= globers["boundaryBin"] + cutoff:
        earlyEventInds.append(i)
    eventCellActivatedFirings[:,i] = activateFiring(avgPeakFiring, preBoundaryFiring)

eventScorer = [-1, -1, 1]
eventAvgClassFiring = [np.mean(eventCellActivatedFirings[:30], axis=0), 
                       np.mean(eventCellActivatedFirings[30:105], axis=0), 
                       np.mean(eventCellActivatedFirings[105:], axis=0)]
eventCellEventness = np.dot(eventScorer, eventAvgClassFiring)
eventnessInds = np.argsort(-eventCellEventness)
eventCellActivatedFirings = eventCellActivatedFirings[:,eventnessInds]

boundaryCellActivatedFirings = np.zeros((globers["nTrials"], boundaryers["nNeurons"]))

for i in range(boundaryers["nNeurons"]):
    recording = cF.loadNeuron(i+1, boundaryers["cellType"])
    preBoundaryFiring = recording[:,:globers["boundaryBin"]]
    peakFiring = recording[:,int(globers["boundaryBin"]+boundaryers["peakFiring"]-2*boundaryers["peakStd"]):int(globers["boundaryBin"] + boundaryers["peakFiring"]+2*boundaryers["peakStd"])]
    avgPeakFiring = np.mean(peakFiring, axis=1)
    if np.argmax(np.mean(recording[30:], axis=0)) > globers["boundaryBin"] + cutoff:
        lateBoundaryInds.append(i)
    boundaryCellActivatedFirings[:,i] = activateFiring(avgPeakFiring, preBoundaryFiring)

boundaryScorer = [-1, 1, 1]
boundaryAvgClassFiring = [np.mean(boundaryCellActivatedFirings[:30], axis=0), 
                       np.mean(boundaryCellActivatedFirings[30:105], axis=0), 
                       np.mean( boundaryCellActivatedFirings[105:], axis=0)]
boundaryCellBoundariness = np.dot(boundaryScorer, boundaryAvgClassFiring)
boundaryInds = np.argsort(-boundaryCellBoundariness)
boundaryCellActivatedFirings = boundaryCellActivatedFirings[:,boundaryInds]

earlyCellFirings = []
lateCellFirings = []

for i in range(boundaryCellActivatedFirings.shape[1]):
    if i in lateBoundaryInds:
        lateCellFirings.append(boundaryCellActivatedFirings[:,i])
    else:
        earlyCellFirings.append(boundaryCellActivatedFirings[:,i])

for i in range(eventCellActivatedFirings.shape[1]):
    if i in earlyEventInds:
        earlyCellFirings.append(eventCellActivatedFirings[:,i])
    else:
        lateCellFirings.append(eventCellActivatedFirings[:,i])

earlyCellFirings = np.array(earlyCellFirings).T
lateCellFirings = np.array(lateCellFirings).T

earlyCellSubFirings = []
lateCellSubFirings = []

# Randomly sample 30 of the SB trials
for i in range(earlyCellFirings.shape[1]):
    sbTrials = np.random.choice(range(30, 105), 30, replace=False)
    earlyCellSubFirings.append(np.append(earlyCellFirings[:30,i], earlyCellFirings[sbTrials,i]))
    earlyCellSubFirings[i] = np.append(earlyCellSubFirings[i], earlyCellFirings[105:,i])

for i in range(lateCellFirings.shape[1]):
    sbTrials = np.random.choice(range(30, 105), 30, replace=False)
    lateCellSubFirings.append(np.append(lateCellFirings[:30,i], lateCellFirings[sbTrials,i]))
    lateCellSubFirings[i] = np.append(lateCellSubFirings[i], lateCellFirings[105:,i])

earlyCellSubFirings = np.array(earlyCellSubFirings).T
lateCellSubFirings = np.array(lateCellSubFirings).T

xAll, yAll = cF.loadEmbeddings()
xAll_NB = xAll[yAll == 0]
xAll_SB = xAll[yAll == 1]
xAll_HB = xAll[yAll == 2]

nTrials = 20

enn_l1Outputs = np.zeros((nTrials, 90, 3))
enn_l2Outputs = np.zeros((nTrials, 90, 3))
mlp_l1Outputs = np.zeros((nTrials, 90, 3))
mlp_l2Outputs = np.zeros((nTrials, 90, 3))

mlpAccs = np.zeros(nTrials)
convThresh = 0.5
for i in range(nTrials):
    xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, 
                                                train_size=globers["trainTestSplit"], 
                                                random_state=globers["randomSeed"]+i)

    xAll_NB_TrialsInds = np.random.choice(range(len(xAll_NB)), 30, replace=False)
    xAll_SB_TrialsInds = np.random.choice(range(len(xAll_SB)), 30, replace=False)
    xAll_HB_TrialsInds = np.random.choice(range(len(xAll_HB)), 30, replace=False)

    xAll_Trials = np.concatenate((xAll_NB[xAll_NB_TrialsInds], 
                                xAll_SB[xAll_SB_TrialsInds], 
                                xAll_HB[xAll_HB_TrialsInds]), axis=0)
    
    mdlENN = lB.main_0(xTrain, yTrain, enners["subconceptSelector"], globers["nHiddenNeurons"])
    mdlMLP = MLPClassifier(hidden_layer_sizes=(globers["nHiddenNeurons"],globers["nHiddenNeurons"]), max_iter=mlpers["maxIter"], 
                           random_state=globers["randomSeed"]+i, activation=mlpers["activation"])
    mdlMLP.fit(xTrain, yTrain)
    mlpAccs[i] = mdlMLP.score(xTest, yTest)

    enn_l1Output = mdlENN.layers[0].compute_output(xAll_Trials, activate=True) # Tanh activation
    enn_l2Output = mdlENN.layers[1].compute_output(enn_l1Output, activate=True) # Tanh activation

    # Flip the sign of the firings rates if the average firing rate on NB is positive
    for i2 in range(3):
        if np.mean(enn_l1Output[:30, i2]) > 0:
            enn_l1Output[:, i2] = -enn_l1Output[:, i2]
        if np.mean(enn_l2Output[:30, i2]) > 0:
            enn_l2Output[:, i2] = -enn_l2Output[:, i2]

    enn_l1Outputs[i] = enn_l1Output
    enn_l2Outputs[i] = enn_l2Output

    if mlpAccs[i] > convThresh:
        mlp_l1Output_unAct = mdlMLP.coefs_[0].T @ xAll_Trials.T + mdlMLP.intercepts_[0].reshape(-1, 1)
        mlp_l1Output = np.tanh(mlp_l1Output_unAct) # Tanh activation
        mlp_l2Output_unAct = mdlMLP.coefs_[1].T @ mlp_l1Output + mdlMLP.intercepts_[1].reshape(-1, 1)
        mlp_l2Output = np.tanh(mlp_l2Output_unAct) # Tanh activation

        mlp_l1Output = mlp_l1Output.T
        mlp_l2Output = mlp_l2Output.T

        for i2 in range(3):
            if np.mean(mlp_l1Output[:30, i2]) > 0:
                mlp_l1Output[:, i2] = -mlp_l1Output[:, i2]
            if np.mean(mlp_l2Output[:30, i2]) > 0:
                mlp_l2Output[:, i2] = -mlp_l2Output[:, i2]

        mlp_l1Outputs[i] = mlp_l1Output
        mlp_l2Outputs[i] = mlp_l2Output



# Remove unconverged networks
mlp_l1Outputs = mlp_l1Outputs[mlpAccs > convThresh]
mlp_l2Outputs = mlp_l2Outputs[mlpAccs > convThresh]
enn_l1Outputs = enn_l1Outputs[mlpAccs > convThresh]
enn_l2Outputs = enn_l2Outputs[mlpAccs > convThresh]

mlpL1Firings = mlp_l1Outputs.reshape(-1, 90, 3)
mlpL2Firings = mlp_l2Outputs.reshape(-1, 90, 3)

ennL1Firings = enn_l1Outputs.reshape(-1, 90, 3)
ennL2Firings = enn_l2Outputs.reshape(-1, 90, 3)

mlpL1Cells = []
mlpL2Cells = []
ennL1Cells = []
ennL2Cells = []

for i in range(ennL1Firings.shape[0]):
    for i2 in range(3):
        mlpL1Cells.append(mlpL1Firings[i,:,i2])
        mlpL2Cells.append(mlpL2Firings[i,:,i2])
        ennL1Cells.append(ennL1Firings[i,:,i2])
        ennL2Cells.append(ennL2Firings[i,:,i2])

mlpL1Cells = np.array(mlpL1Cells)
mlpL2Cells = np.array(mlpL2Cells)
ennL1Cells = np.array(ennL1Cells)
ennL2Cells = np.array(ennL2Cells)

mlpL1Cells = np.reshape(mlpL1Cells, (-1, 90))
mlpL2Cells = np.reshape(mlpL2Cells, (-1, 90))
ennL1Cells = np.reshape(ennL1Cells, (-1, 90))
ennL2Cells = np.reshape(ennL2Cells, (-1, 90))

# Sort the cells by average firing rate [Update this for Milo]
# Need to put Boundary like cells first, then Event like cells, then the rest
def sortCells(_cells):
    _threshold = 1.5
    _boundaryScorer = [-1, 1, 1]
    _eventScorer = [-1, -1, 1]
    _boundaryCells = []
    _eventCells = []
    _otherCells = []

    for i in range(_cells.shape[0]):
        boundaryScore = np.dot(_boundaryScorer, [np.mean(_cells[i,:30]), np.mean(_cells[i,30:60]), np.mean(_cells[i,60:])])
        eventScore = np.dot(_eventScorer, [np.mean(_cells[i,:30]), np.mean(_cells[i,30:60]), np.mean(_cells[i,60:])])
        if boundaryScore > eventScore and boundaryScore > _threshold:
            _boundaryCells.append(_cells[i])
        elif eventScore > boundaryScore and eventScore > _threshold:
            _eventCells.append(_cells[i])
        else:
            _otherCells.append(_cells[i])

    _boundaryCells = np.array(_boundaryCells)
    _boundaryCells = np.reshape(_boundaryCells, (-1, 90))

    _eventCells = np.array(_eventCells)
    _eventCells = np.reshape(_eventCells, (-1, 90))

    _otherCells = np.array(_otherCells)
    _otherCells = np.reshape(_otherCells, (-1, 90))

    # Sort within the boundary and event cells
    _boundaryCells = _boundaryCells[np.argsort(-np.mean(_boundaryCells[:,30:], axis=1))]
    _eventCells = _eventCells[np.argsort(-np.mean(_eventCells[:,60:], axis=1))]
    _otherCells = _otherCells[np.argsort(-np.mean(_otherCells, axis=1))]

    print("Boundary Cells: {}".format(_boundaryCells.shape))
    print("Event Cells: {}".format(_eventCells.shape))
    print("Other Cells: {}".format(_otherCells.shape))

    # Append all cells
    _allCells = np.concatenate((_boundaryCells, _eventCells, _otherCells), axis=0)
    print("All Cells: {}".format(_allCells.shape))

    return _allCells

ennL1Cells = sortCells(ennL1Cells)
ennL2Cells = sortCells(ennL2Cells)    

mlpL1Cells = sortCells(mlpL1Cells)
mlpL2Cells = sortCells(mlpL2Cells)

figWidth = 7
figHeight = 2.5
fig, ax = plt.subplots(1, 6, figsize=(figWidth, figHeight), tight_layout=True, sharey=True)
_aspect=1

ax[0].imshow(earlyCellSubFirings, cmap="gray_r", 
             aspect=_aspect, 
             vmin=0, vmax=1, interpolation="none")
ax[1].imshow(ennL1Cells.T, cmap="gray_r", 
    aspect=_aspect, 
    vmin=-1, vmax=1, interpolation="none")
ax[2].imshow(mlpL1Cells.T, cmap="gray_r",
    aspect=_aspect, vmin=-1, vmax=1, interpolation="none")

ax[3].imshow(lateCellSubFirings, cmap="gray_r",
                aspect=_aspect, vmin=0, vmax=1, interpolation="none")
ax[4].imshow(ennL2Cells.T, cmap="gray_r",
    aspect=_aspect, vmin=-1, vmax=1, interpolation="none")
ax[5].imshow(mlpL2Cells.T, cmap="gray_r",
    aspect=_aspect, vmin=-1, vmax=1, interpolation="none")

titles = ["MTL", "ENN", "Backprop"]
for i in range(6):
    ax[i].set_yticks([0,30,60,90], labels=[])
    ax[i].axhspan(0, 30, color=cC.conceptColors[0], alpha=0.1)
    ax[i].axhspan(30, 30 + 30, color=cC.conceptColors[1], alpha=0.1)
    ax[i].axhspan(30 + 30, 30 + 30 + 30, color=cC.conceptColors[2], alpha=0.1)
    ax[i].set_ylim(89, 0)
    ax[i].set_xticks([], labels=[])
    ax[i].set_title(titles[i%3])

ax[1].set_xlabel("Early / Layer 1 Neurons")
ax[4].set_xlabel("Late / Layer 2 Neurons")

fig.tight_layout()
fig.savefig(logMan.mediaDir + "/Fig2d.png",
            transparent=True, 
            dpi=300)
plt.close(fig)
"""

# Figure 2e: Neuron Average Firing
# Functions
def activateFiring(_avgPeakFiring, _backgroundFiring):
    _scaleFactor = 0.5
    _xOffset = 1.5
    _steepness = 0.1 / np.std(np.mean(_backgroundFiring, axis=1))
    _yOffset = 0.5

    _normalizedFiring = _avgPeakFiring / np.mean(_backgroundFiring)

    return _scaleFactor * np.tanh(_steepness * (_normalizedFiring - _xOffset)) + _yOffset

def processNeurons(_nNeurons, _cellType, _peakFiring, _peakStd, globers):
    _activatedFirings = np.zeros((globers["nTrials"], _nNeurons))
    
    for i in range(_nNeurons):
        _recording = cF.loadNeuron(i + 1, _cellType)
        _preBoundaryFiring = _recording[:, :globers["boundaryBin"]]
        _peakFiringRange = _recording[:, int(globers["boundaryBin"] + _peakFiring - 2*_peakStd):int(globers["boundaryBin"] + _peakFiring + 2*_peakStd)]
        _avgPeakFiring = np.mean(_peakFiringRange, axis=1)
        _activatedFirings[:, i] = activateFiring(_avgPeakFiring, _preBoundaryFiring)
        
    return _activatedFirings

def computeStats(_activatedFirings):
    avgNB = np.mean(_activatedFirings[:30], axis=0)
    avgSB = np.mean(_activatedFirings[30:30 + 75], axis=0)
    avgHB = np.mean(_activatedFirings[30 + 75:], axis=0)

    stdNB = np.std(_activatedFirings[:30], axis=0)
    stdSB = np.std(_activatedFirings[30:30 + 75], axis=0)
    stdHB = np.std(_activatedFirings[30 + 75:], axis=0)
    
    return [avgNB, avgSB, avgHB], [stdNB, stdSB, stdHB]

eventers = {
    "nNeurons": 36,
    "cellType": "HB",
    "peakFiring": 30.1,
    "peakStd": 5.5,
}

boundaryers = {
    "nNeurons": 42,
    "cellType": "B",
    "peakFiring": 19.7,
    "peakStd": 4.9
}

globers = {
    "randomSeed": 42,
    "nTrials": 135,
    "nBins": 150,
    "msPerBin": 10,
    "boundaryBin": 50,
    "nHiddenNeurons": 3,
    "trainTestSplit": 0.5,
}

eventCellActivatedFirings = processNeurons(eventers["nNeurons"], eventers["cellType"], eventers["peakFiring"], eventers["peakStd"], globers)
boundaryCellActivatedFirings = processNeurons(boundaryers["nNeurons"], boundaryers["cellType"], boundaryers["peakFiring"], boundaryers["peakStd"], globers)

eventCellActivatedFiringsAvg, eventCellActivatedFiringsStd = computeStats(eventCellActivatedFirings)
boundaryCellActivatedFiringsAvg, boundaryCellActivatedFiringsStd = computeStats(boundaryCellActivatedFirings)

xAll, yAll = cF.loadEmbeddings()
xAll_NB = xAll[yAll == 0]
xAll_SB = xAll[yAll == 1]
xAll_HB = xAll[yAll == 2]

nNetworks = 20
enn_l1Outputs = np.zeros((nNetworks, 90, 3))
enn_l2Outputs = np.zeros((nNetworks, 90, 3))
mlp_l1Outputs = np.zeros((nNetworks, 90, 3))
mlp_l2Outputs = np.zeros((nNetworks, 90, 3))

mlpAccs = np.zeros(nNetworks)
convThresh = 0.5

for i in range(nNetworks):
    xTrain, xTest, yTrain, yTest = train_test_split(xAll, yAll, 
                                                train_size=globers["trainTestSplit"], 
                                                random_state=42+i)

    mdlMLP = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=10000, random_state=42+i, activation="tanh")
    mdlMLP.fit(xTrain, yTrain)
    mlpAccs[i] = mdlMLP.score(xTest, yTest)
    mdlENN = lB.main_0(xTrain, yTrain, 0, globers["nHiddenNeurons"])

    xAll_NB_TrialsInds = np.random.choice(range(len(xAll_NB)), 30, replace=False)
    xAll_SB_TrialsInds = np.random.choice(range(len(xAll_SB)), 30, replace=False)
    xAll_HB_TrialsInds = np.random.choice(range(len(xAll_HB)), 30, replace=False)

    xAll_Trials = np.concatenate((xAll_NB[xAll_NB_TrialsInds], 
                                xAll_SB[xAll_SB_TrialsInds], 
                                xAll_HB[xAll_HB_TrialsInds]), axis=0)

    enn_l1Output = mdlENN.layers[0].compute_output(xAll_Trials, activate=True) # Tanh activation
    enn_l2Output = mdlENN.layers[1].compute_output(enn_l1Output, activate=True) # Tanh activation

    # Flip the sign of the firings rates if the average firing rate on NB is positive
    for i2 in range(3):
        if np.mean(enn_l1Output[:30, i2]) > 0:
            enn_l1Output[:, i2] = -enn_l1Output[:, i2]
        if np.mean(enn_l2Output[:30, i2]) > 0:
            enn_l2Output[:, i2] = -enn_l2Output[:, i2]

    enn_l1Outputs[i] = enn_l1Output
    enn_l2Outputs[i] = enn_l2Output

    if mlpAccs[i] > convThresh:
        mlp_l1Output_unAct = mdlMLP.coefs_[0].T @ xAll_Trials.T + mdlMLP.intercepts_[0].reshape(-1, 1)
        mlp_l1Output = np.tanh(mlp_l1Output_unAct) # Tanh activation
        mlp_l2Output_unAct = mdlMLP.coefs_[1].T @ mlp_l1Output + mdlMLP.intercepts_[1].reshape(-1, 1)
        mlp_l2Output = np.tanh(mlp_l2Output_unAct) # Tanh activation

        mlp_l1Output = mlp_l1Output.T
        mlp_l2Output = mlp_l2Output.T

        for i2 in range(3):
            if np.mean(mlp_l1Output[:30, i2]) > 0:
                mlp_l1Output[:, i2] = -mlp_l1Output[:, i2]
            if np.mean(mlp_l2Output[:30, i2]) > 0:
                mlp_l2Output[:, i2] = -mlp_l2Output[:, i2]

        mlp_l1Outputs[i] = mlp_l1Output
        mlp_l2Outputs[i] = mlp_l2Output

# Remove unconverged networks
mlp_l1Outputs = mlp_l1Outputs[mlpAccs > convThresh]
mlp_l2Outputs = mlp_l2Outputs[mlpAccs > convThresh]
enn_l1Outputs = enn_l1Outputs[mlpAccs > convThresh]
enn_l2Outputs = enn_l2Outputs[mlpAccs > convThresh]

enn_l1AvgNBFiring = np.mean(enn_l1Outputs[:,:30,:], axis=1)
enn_l1AvgSBFiring = np.mean(enn_l1Outputs[:,30:30+30,:], axis=1)
enn_l1AvgHBFiring = np.mean(enn_l1Outputs[:,30+30:,:], axis=1)

enn_l2AvgNBFiring = np.mean(enn_l2Outputs[:,:30,:], axis=1)
enn_l2AvgSBFiring = np.mean(enn_l2Outputs[:,30:30+30,:], axis=1)
enn_l2AvgHBFiring = np.mean(enn_l2Outputs[:,30+30:,:], axis=1)

ennl1StdNBFiring = np.std(enn_l1Outputs[:,:30,:], axis=1)
ennl1StdSBFiring = np.std(enn_l1Outputs[:,30:30+30,:], axis=1)
ennl1StdHBFiring = np.std(enn_l1Outputs[:,30+30:,:], axis=1)

ennl2StdNBFiring = np.std(enn_l2Outputs[:,:30,:], axis=1)
ennl2StdSBFiring = np.std(enn_l2Outputs[:,30:30+30,:], axis=1)
ennl2StdHBFiring = np.std(enn_l2Outputs[:,30+30:,:], axis=1)

mlp_l1AvgNBFiring = np.mean(mlp_l1Outputs[:,:30,:], axis=1)
mlp_l1AvgSBFiring = np.mean(mlp_l1Outputs[:,30:30+30,:], axis=1)
mlp_l1AvgHBFiring = np.mean(mlp_l1Outputs[:,30+30:,:], axis=1)

mlp_l2AvgNBFiring = np.mean(mlp_l2Outputs[:,:30,:], axis=1)
mlp_l2AvgSBFiring = np.mean(mlp_l2Outputs[:,30:60,:], axis=1)
mlp_l2AvgHBFiring = np.mean(mlp_l2Outputs[:,30+30:,:], axis=1)

mlp_l1StdNBFiring = np.std(mlp_l1Outputs[:,:30,:], axis=1)
mlp_l1StdSBFiring = np.std(mlp_l1Outputs[:,30:60,:], axis=1)
mlp_l1StdHBFiring = np.std(mlp_l1Outputs[:,30+30:,:], axis=1)

mlp_l2StdNBFiring = np.std(mlp_l2Outputs[:,:30,:], axis=1)
mlp_l2StdSBFiring = np.std(mlp_l2Outputs[:,30:60,:], axis=1)
mlp_l2StdHBFiring = np.std(mlp_l2Outputs[:,30+30:,:], axis=1)

# Plot the results
figWidth = 8
figHeight = 4
fig, ax = plt.subplots(1, 3, figsize=(figWidth, figHeight), subplot_kw={'projection': '3d'})

ax[0].scatter(eventCellActivatedFiringsAvg[0], eventCellActivatedFiringsAvg[1], eventCellActivatedFiringsAvg[2], color=cC.cellColors["Event"], label='Event Cell', alpha=1)
ax[0].scatter(boundaryCellActivatedFiringsAvg[0], boundaryCellActivatedFiringsAvg[1], boundaryCellActivatedFiringsAvg[2], color=cC.cellColors["Boundary"], label='Boundary Cell', alpha=1)
ax[0].legend(bbox_to_anchor=(0., 2), loc='upper left', ncols=2)

ax[1].scatter(enn_l1AvgNBFiring, enn_l1AvgSBFiring, enn_l1AvgHBFiring, color='blue', label='Layer 1', alpha=1)
ax[1].scatter(enn_l2AvgNBFiring, enn_l2AvgSBFiring, enn_l2AvgHBFiring, color='red', label='Layer 2', alpha=1)
#ax[1].legend(bbox_to_anchor=(0.5, 2), loc='upper left', ncols=2)

ax[2].scatter(mlp_l1AvgNBFiring, mlp_l1AvgSBFiring, mlp_l1AvgHBFiring, color='blue', label='Layer 1', alpha=1)
ax[2].scatter(mlp_l2AvgNBFiring, mlp_l2AvgSBFiring, mlp_l2AvgHBFiring, color='red', label='Layer 2', alpha=1)
ax[2].legend(bbox_to_anchor=(-0.5, 2), loc='upper left', ncols=2)

ax[0].set_title("MTL")
ax[1].set_title("ENN")
ax[2].set_title("Backprop")

ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].set_zlim([0, 1])

ax[0].set_xticks([0, 0.5, 1])
ax[0].set_yticks([0, 0.5, 1])
ax[0].set_zticks([0, 0.5, 1])

for i in range(3):
    ax[i].set_xlabel("{} Firing".format(r'$\overline{NB}$'))
    ax[i].set_ylabel("{} Firing".format(r'$\overline{SB}$'))
    ax[i].set_zlabel("{} Firing".format(r'$\overline{HB}$'))

for i in range(1,3):
    ax[i].set_xlim([-1, 1])
    ax[i].set_ylim([-1, 1])
    ax[i].set_zlim([-1, 1])

    ax[i].set_xticks([-1, 0, 1])
    ax[i].set_yticks([-1, 0, 1])
    ax[i].set_zticks([-1, 0, 1])

fig.subplots_adjust(wspace=0.75)
fig.savefig(logMan.mediaDir + "/Fig2e.png",
            transparent=True, 
            dpi=300)
plt.close(fig)