#!/usr/bin/env python
"""This script runs the ENN algorithm on the MNIST dataset"""

#Imports
import numpy as np
from enn.train_enn import train_enn

import os
import copy
import pandas as pd
import random
from datetime import date
dir_path = os.path.dirname(os.path.realpath(__file__))

def set_parameters():
    """Set parameters for training ENN on CA"""
    global parameters
    parameters = {
        'cross_val_fold': 0,
        'num_subclasses': 3,
        'alternative_subclasses': 1,
        'svm_cost_1': 15,
        'margin_ratio': np.inf,
        'svm_multiplier_1': 8,
        'svm_multiplier_2': 8,
        'svm_cost_2': 1000,
        'misclass_tolerance': 0,
        'concept_certainty': 1,
        'verbose': False,
        'regression': False,
        'convolutional': False
    }

def set_parameters_explicit(alternative_subclasses, num_subclasses, margin_ratio=np.inf):
    """Set parameters for training ENN on CA"""
    global parameters
    parameters = {
        'cross_val_fold': 0,
        'num_subclasses': num_subclasses,
        'alternative_subclasses': alternative_subclasses,
        'svm_cost_1': 15,
        'margin_ratio': margin_ratio,
        'svm_multiplier_1': 8,
        'svm_multiplier_2': 8,
        'svm_cost_2': 1000,
        'misclass_tolerance': 0,
        'concept_certainty': 1,
        'verbose': True,
        'regression': False,
        'convolutional': False
    }


def evaluate_network(network, deliberate=False):
    """Load test data and compute test error, with option for dENN deliberation"""
    test_data = np.loadtxt('MNIST_test_images.csv', delimiter=',')
    test_labels = np.loadtxt('MNIST_test_labels.csv')
    network.compute_error(test_data, test_labels, deliberate=deliberate)
    
def evaluate_network_0(network, test_inputs, test_labels, deliberate=False):
    """Load test data and compute test error, with option for dENN deliberation"""
    err = network.compute_error(test_inputs, test_labels, deliberate=deliberate)
    acc = 1 - err[0]
    return acc


def get_job_id():
    """Returns the job id from the HPC"""
    if 'SLURM_JOB_ID' in os.environ:
        return os.environ['SLURM_JOB_ID']
    else:
        return 0

def main(num_samples, suffix):
    """The main function"""
    set_parameters(num_samples)
    network = train_network(num_samples)
    today = date.today()
    today = today.strftime("%m-%d-%Y")
    if network: #a network might return None, for example during cross-validation runs
        #Print the network size
        print('Size:', end=' ')
        for layer in network.layers:
            print(layer.num_nodes(), end=' ')
            if layer.convolutional:
                print('(', layer.win_size, ')', end=' ')
        print()

        #Print the number of samples used by differentiae as support vectors
        num_sv = network.layers[len(network.layers)-3].num_sv()
        print('Support Vectors: ', int(num_sv), 'or', round(100*num_sv/network.parameters['training_samples'], 2), '%')
        
        #Evaluate network performance without and with deliberation
        evaluate_network(network, deliberate=False)
        #evaluate_network(network, deliberate=True)

        #Save the network with appropriate suffix
        save_name_0 = "mnist_enn"
        save_name_1 = str(num_samples)
        save_name_2 = str(today) + "_" + str(suffix)
        network.save_network(save_name_0, save_name_1, save_name_2)

def main_0(img, labels, alternative_subclasses, num_subclasses, margin_ratio=np.inf):
    """The main function"""
    print("Training network with " + str(len(labels)) + " images")
    set_parameters_explicit(alternative_subclasses, num_subclasses, margin_ratio)

    network = train_enn(img, labels, parameters)
    
    return network

