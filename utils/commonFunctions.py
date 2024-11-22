# Path: release/utils/commonFunctions.py

import numpy as np
import scipy
import matplotlib.pyplot as plt

def loadEmbeddings(projDir="/project/greencenter/Lin_lab/s181641/bdENS/", branch="develop", verbose=True):
    _xAll = np.load(projDir + branch + "/inputs/boundaryEmbeddings.npy")
    _yAll = np.load(projDir + branch + "/inputs/boundaryLabels.npy").astype(int)
    if verbose:
        print("Loaded {} inputs and {} labels".format(len(_xAll), len(_yAll)))
        print("Input dimensionality: {}".format(_xAll.shape[1]))
        print("Labels are: {}".format(np.unique(_yAll)))
    return _xAll, _yAll

def getHiddenLayerFirings(_mdl, _xAll, mdlType="enn", activate=True):
    if mdlType == "enn":
        _l1Outputs_unAct = _mdl.layers[0].compute_output(_xAll, activate=False)
        _l1Outputs = _mdl.layers[0].compute_output(_xAll, activate=True) # Tanh activation
        _l2Outputs_unAct = _mdl.layers[1].compute_output(_l1Outputs, activate=False)
        _l2Outputs = _mdl.layers[1].compute_output(_l1Outputs, activate=True) # Tanh activation
    elif mdlType == "mlp":
        _l1Outputs_unAct = _mdl.coefs_[0].T @ _xAll.T + _mdl.intercepts_[0].reshape(-1, 1)
        _l1Outputs = 1 / (1 + np.exp(-_l1Outputs_unAct)) # Logistic activation
        _l2Outputs_unAct = _mdl.coefs_[1].T @ _l1Outputs + _mdl.intercepts_[1].reshape(-1, 1)
        _l2Outputs = 1 / (1 + np.exp(-_l2Outputs_unAct)) # Logistic activation

    if activate:
        return _l1Outputs, _l2Outputs
    else:
        return _l1Outputs_unAct, _l2Outputs_unAct
    
def getTruePosInds(_mdl, _x, _y, mdlType="enn", verbose="True"):
    if mdlType == "enn":
        _preds = np.argmax(_mdl.compute_output(_x), axis=1)
    elif mdlType == "mlp":
        _preds = _mdl.predict(_x)
    _tpInds = np.where(_preds == _y)[0]
    if verbose:
        print("True positive rate: {}/{}".format(len(_tpInds), len(_x)))
    return _tpInds

def loadNeuron(_nNeuron, _cellType, projDir="/project/greencenter/Lin_lab/s181641/bdENS/", branch="develop"):
    """
    Load neuron recordings from fJie dataset
        source: https://www.nature.com/articles/s41593-022-01020-w
    args:
            neuron_no: int, neuron number
                valid neurons are: 1-42 for boundary cells, 1-36 for event cells
            cell_type: str, 'B' for boundary cell, 'HB' for event cell
                
        returns:
            spks_per_trial: numpy array, shape (135, 150), spikes per trial
    """

    # Check for valid cell_type and neuron_no
    if _cellType not in ['B', 'HB']:
        raise ValueError('Invalid cell_type: {}'.format(_cellType))
    if _cellType == 'B' and _nNeuron not in range(1, 43):
        print("For Boundary cells, neuron_no should be in range(1, 43)")
        raise ValueError('Invalid neuron_no: {} for Boundary Cell'.format(_nNeuron))
    if _cellType == 'HB' and _nNeuron not in range(1, 37):
        print("For Event cells, neuron_no should be in range(1, 37)")
        raise ValueError('Invalid neuron_no: {} for Event Cell'.format(_nNeuron))

    # Load recording
    recording = scipy.io.loadmat(projDir + branch + '/inputs/fJie/{}Cell{}_BAligned.mat'.format(_cellType, _nNeuron))

    # Return spikes per trial
    return recording['spks_per_trial']

exampleRecording = loadNeuron(1, 'B')

def updatePlotrcs(font='monospace'):
    # Set default plot parameters
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'axes.labelsize': 10})
    plt.rcParams.update({'axes.titlesize': 12})
    plt.rcParams.update({'xtick.labelsize': 10})
    plt.rcParams.update({'ytick.labelsize': 10})
    plt.rcParams.update({'legend.fontsize': 10})
    plt.rcParams.update({'figure.titlesize': 12})

    if font == 'monospace':
        # Set font to monospace
        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.monospace'] = ['Courier New'] + plt.rcParams['font.monospace']
    elif font == 'serif':
        # Set font to times new roman
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Turn off right and top spine
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

# Helper function to convert hex to RGB
def hex2rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Helper function to convert RGB to hex
def rgb2hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(*rgb_color)

# Generic gradient making function
def genColorGradient(_startHex, _endHex, _nSteps):
    # Convert start and end colors to RGB
    _startRGB = np.array(hex2rgb(_startHex))
    _endRGB = np.array(hex2rgb(_endHex))
    
    # Create a gradient by interpolating between the start and end RGB colors
    _gradient = np.linspace(_startRGB, _endRGB, _nSteps, dtype=int)
    
    # Convert each RGB value back to hex
    _hexGradient = [rgb2hex(tuple(_rgb)) for _rgb in _gradient]
    
    return _hexGradient

# Extract trials
def genTrials(_xAll, _yAll, classTrials=[30, 75, 30]):
    _nb = _xAll[_yAll == 0]
    _sb = _xAll[_yAll == 1]
    _hb = _xAll[_yAll == 2]
    _nbInds = np.random.choice(range(len(_nb)), classTrials[0], replace=False)
    _sbInds = np.random.choice(range(len(_sb)), classTrials[1], replace=False)
    _hbInds = np.random.choice(range(len(_hb)), classTrials[2], replace=False)
    _allTrials = np.concatenate((_nb[_nbInds], 
                                  _sb[_sbInds], 
                                  _hb[_hbInds]), axis=0)
    return _allTrials