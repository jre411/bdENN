"""A Network object that contains an ENN"""

from enn.layer import Layer
from enn.subclass import Subclass
import numpy as np
import os
#import matplotlib.pyplot as plt

class Network():
    """A structure to store an ENN"""
    def __init__(self, parameters):
        self.layers = []
        self.parameters = parameters
        self.regression = False
        self.multilabel = False
        if parameters:
            if parameters['regression']:
                self.regression = True
        self.subclasses = []
        self.convolutional = False

    def copy(self):
        """Make a complete copy of itself"""
        network = Network(self.parameters.copy())
        network.regression = self.regression
        network.multilabel = self.multilabel
        network.convolutional = self.convolutional
        for layer in self.layers:
            network.add_layer(layer.copy())
        return network

    def compute_output(self, x_test, deliberate=False, return_all_firing=False, del_strictness=2, last_activation=True):
        """Returns the output of the network for a given x; there are several methods below for deliberation, but when
        deliberate=True then the standard method is run"""

        if not deliberate:
            #The standard method, without deliberation, of computing the network's output
            nodes = x_test
            for l_i,layer in enumerate(self.layers):
                nodes = layer.compute_output(nodes, activate=(l_i<len(self.layers)-1 or last_activation))
            if len(nodes.shape)==1:
                nodes = np.reshape(nodes, (1, -1))
            return_all_firing = False
        else:
            #Deliberate by changing biases of the subconcept neurons
            diff_output = x_test
            for layer_i in range(len(self.layers)-2):
                diff_output = self.layers[layer_i].compute_output(diff_output)
            subc_output = self.layers[-2].compute_output(diff_output)
            conc_output = self.layers[-1].compute_output(subc_output)
            nodes = np.zeros(conc_output.shape)

            #Do deliberation on each sample
            prob_cutoff = .5
            is_tanh = False
            if self.layers[-2].activation_function=='tanh':
                is_tanh = True
                prob_cutoff = 0
            for sample in range(x_test.shape[0]):
                remaining_outputs = np.arange(self.layers[-1].num_nodes())
                remaining_subc = np.where(np.array(self.layers[-2].levels)==0)[0]#np.arange(len(self.subclasses))
                layer1 = self.layers[-2].copy()
                for guess_num in range(min(deliberate,self.layers[-1].num_nodes()-1)):
                    probs = np.sort(conc_output[sample, remaining_outputs])[::-1]
                    if probs[0]<probs[1]*del_strictness or probs[0]<.5:
                        best_prob = probs[0]
                        best_conc_output = conc_output[sample, :].copy()
                        best_subc_output = subc_output[sample, :].copy()
                        pre_subc_output = layer1.compute_output(diff_output[sample, :], activate=False)
                        if np.max(subc_output[sample, remaining_subc])>prob_cutoff: #many options, so decrease bias for all
                            del_change = -np.max(pre_subc_output)*.01
                        else:
                            ref = np.sort(pre_subc_output).flatten()
                            s = min(5, layer1.num_nodes())
                            del_change = -ref[-(s+1)]*.01
                        
                        for _ in range(500):
                            layer1.biases += del_change
                            subc_output_temp = layer1.compute_output(diff_output[sample, :])
                            #conc_output_temp = self.layers[-1].compute_output(subc_output_temp, remaining_outputs=remaining_outputs)
                            conc_output_temp = self.layers[-1].compute_output(subc_output_temp)
                            probs = np.sort(conc_output_temp[0][remaining_outputs].flatten())[::-1]
                            if probs[0]>best_prob:
                                best_conc_output = conc_output_temp.copy()
                                best_subc_output = subc_output_temp.copy()
                                best_prob = probs[0]
                            if del_strictness<=2:
                                if probs[0]>probs[1]*del_strictness:
                                    break
                            else:
                                subc_prob = np.sort(subc_output_temp.flatten())[::-1]
                                if is_tanh:
                                    subc_prob += 1
                                    subc_prob /= 2
                                if subc_prob[0]>subc_prob[1]*2:
                                    best_conc_output = conc_output_temp.copy()
                                    best_subc_output = subc_output_temp.copy()
                                    break
                        subc_output[sample, :] = best_subc_output
                        conc_output[sample, :] = best_conc_output
                    if deliberate>1:
                        #An optional, different type of deliberation that allows the ENN to realize it got its first guess wrong,
                        #artificially silence the related subconcepts, and then recalculate the concept outputs
                        guess = np.argmax(conc_output[sample, :])
                        nodes[sample, guess] = .5**guess_num
                        remaining_outputs = np.delete(remaining_outputs, np.where(remaining_outputs==guess))
                        for sc_i in range(len(remaining_subc)):
                            if self.subclasses[remaining_subc[sc_i]].y_class[0] == self.parameters['classes'][guess]:
                                remaining_subc[sc_i] = -1
                        remaining_subc = np.delete(remaining_subc, np.where(remaining_subc<0))
                        conc_output[sample, guess] = 0
                        conc_output[sample, :] /= np.sum(conc_output[sample, :])
                    else:
                        nodes[sample, :] = conc_output[sample, :]
        """
        #Alternative deliberation method
            diff_output = self.layers[0].compute_output(x_test)
            subc_output = self.layers[1].compute_output(diff_output)
            conc_output = self.layers[2].compute_output(subc_output)                

            for sample in range(x_test.shape[0]):
                probs = np.sort(conc_output[sample, :])[::-1]
                if probs[0]>probs[1]*2:
                    continue
                layer1 = self.layers[1].copy()
                remaining_outputs_temp = remaining_outputs.copy()
                for _ in range(conc_output.shape[1]-1):
                    to_remove = remaining_outputs_temp[np.argsort(conc_output[sample, remaining_outputs_temp])[0]]
                    sc_list = []
                    for sc_i, sc in enumerate(self.subclasses):
                        if sc.y_class[0] == self.parameters['classes'][0][to_remove]:
                            sc_list.append(sc_i)
                    diff_list = []
                    for sc_i in sc_list:
                        diff_list.extend(self.layers[0].get_subclass_diff(sc_i))
                    diff_list = np.unique(diff_list)
                    layer1.biases += np.sum(np.multiply(layer1.midpoints[diff_list, :], layer1.weights[diff_list, :]), axis=0)
                    layer1.weights[diff_list, :] = 0
                    subc_output_temp = layer1.compute_output(diff_output[sample, :])
                    remaining_outputs_temp = np.delete(remaining_outputs_temp, np.where(remaining_outputs_temp == to_remove))
                    conc_output_temp = self.layers[2].compute_output(subc_output_temp, remaining_outputs=remaining_outputs_temp)
                    probs = np.sort(conc_output_temp[0])[::-1]
                    if probs[0]>probs[1]*2:
                        break
                conc_output[sample, :] = conc_output_temp[0]
            nodes = conc_output
        """

        if return_all_firing:
            return [diff_output, subc_output, conc_output]
        else:     
            return nodes
    
    def add_layer(self, new_layer):
        """Appends a layer to the network"""
        self.layers.append(new_layer)
        if new_layer.multilabel:
            self.multilabel = True
        if new_layer.convolutional:
            self.convolutional = True
    
    def delete_node(self, layer, node):
        """Deletes a node from a given layer of the network"""
        self.layers[layer].del_node(node)
        self.layers[layer+1].weights = np.delete(self.layers[layer+1].weights, node, axis=0)

    def get_labels(self, output):
        """Assigns class labels to each output"""
        if not self.multilabel:
            labels = np.zeros(output.shape)
            class_order = np.argsort(-output+np.random.rand(output.shape[0],output.shape[1])*1e-11, axis=1)
            class_order = np.argsort(-output, axis=1)
            for i, l in enumerate(self.parameters['classes']):
                np.place(labels, class_order==i, l)
        else:
            labels = np.zeros((output.shape[0], len(self.layers[-1].label_lengths)))
            cum_sum = 0
            for label,length in enumerate(self.layers[-1].label_lengths):
                output_ind = (np.arange(length)+cum_sum).astype(int)
                label_ind = np.argmax(output[:, output_ind], axis=1)
                for sample in range(len(label_ind)):
                    labels[sample, label] = self.parameters['classes'][label][label_ind[sample]]
                cum_sum += length
        return labels
    
    def predict(self, x_test, deliberate=False):
        """Predict output labels for test set"""
        output = self.compute_output(x_test, deliberate=deliberate)
        labels = self.get_labels(output)
        return labels[:, 0]

    def compute_error_from_output(self, output, y_test):
        """Calculates the predictive error of a set of outputs"""

        if self.regression:
            err = np.sqrt(np.mean((output-y_test)**2))
        elif self.multilabel:
            labels = self.get_labels(output)
            err = np.zeros(2)
            err[0] = np.mean(labels!=y_test)
            err[1] = np.mean(np.sum(labels==y_test, axis=1)!=y_test.shape[1])
        else:
            labels = self.get_labels(output)
            acc = np.zeros(len(self.parameters['classes']))
            cumulative_acc = 0
            for i in range(labels.shape[1]): # pylint: disable=E1136  # pylint/issues/3139
                cumulative_acc += np.sum(np.equal(labels[:,i], y_test))
                acc[i] = cumulative_acc
            
            err = 1-acc/len(y_test)
            

        return err

    def compute_error(self, x_test, y_test, print_error=True, per_class=False, deliberate=False, del_strictness=2, output=None):
        """Calculates the predictive error of the network"""

        if output is None:
            output = self.compute_output(x_test, deliberate=deliberate, del_strictness=del_strictness)
        if per_class:
            unique_y = np.unique(y_test)
            err = np.zeros(len(unique_y))
            for y_class in range(len(unique_y)):
                class_indices = [i for i in range(len(y_test)) if y_test[i]==unique_y[y_class]]
                err[y_class] = self.compute_error_from_output(output[class_indices, :], y_test[class_indices])[0]
        else:
            err = self.compute_error_from_output(output, y_test)
            if isinstance(err, float):
                err = [err]

        if print_error:
            print('Error:', end=' ')
            for e in err:
                print(round(e,4), end=' ')
            print()

        return err
    
    def save_network(self, filename, jobid=0, network_id=0):
        """Saves the network"""
        while True:
            #Save the network with a unique number
            trial_filename = filename + '_' + str(jobid) + '_' + str(0) + '.npz'
            if os.path.isfile(trial_filename):
                jobid += 1
            else:
                break
        full_filename = filename + '_' + str(jobid) + '_' + str(network_id) + '.npz'

        #Convolutional networks need a different method to save them
        if self.convolutional:
            self.save_convolutional_network(filename + '_' + str(jobid) + '_')
            print('Saved as convolutional ENN at ' + filename + '_' + str(jobid) + '_')
            return

        #Get all of the necessary information and then store
        differentia_layer = np.concatenate((self.layers[0].weights, self.layers[0].biases.reshape(1, -1)))
        subclass_layer = np.concatenate((self.layers[1].weights, self.layers[1].biases.reshape(1, -1)))
        concept_layer = np.concatenate((self.layers[2].weights, self.layers[2].biases.reshape(1, -1)))
        activations = [l.activation_function for i,l in enumerate(self.layers)]
        sum_dual_coef_diff = self.layers[0].sum_dual_coef
        sum_dual_coef_sub = self.layers[1].sum_dual_coef
        
        num_points = 0
        for sc in self.subclasses:
            num_points += len(sc.points)
        if self.multilabel:
            subclass_points = np.zeros((num_points,))
            subclass_labels = np.zeros((len(self.subclasses), self.layers[2].num_nodes()))
            for sc_i, sc in enumerate(self.subclasses):
                subclass_points[sc.points] = sc_i
                subclass_labels[sc_i] = sc.y_class[0]
        else:
            subclass_points = np.zeros((num_points,))
            subclass_labels = np.zeros((len(self.subclasses), 2))
            for sc_i, sc in enumerate(self.subclasses):
                subclass_points[sc.points] = sc_i
                subclass_labels[sc_i] = sc.y_class
        
        if self.layers[2].multilabel:
            label_lengths = self.layers[2].label_lengths
        else:
            label_lengths = self.layers[2].num_nodes()
        
        symbolic = [l.symbolic for l in self.layers]
        
        support_vectors = np.zeros((0,2))
        for i, svs in enumerate(self.layers[0].support_vectors):
            diff_svs = np.hstack((np.ones((len(svs), 1))*i, svs.reshape(-1, 1)))
            support_vectors = np.vstack((support_vectors, diff_svs))

        np.savez(full_filename, differentia_layer=differentia_layer, subclass_layer=subclass_layer, concept_layer=concept_layer, activations=activations,
            differentia_midpoints=self.layers[0].midpoints, subclass_midpoints=self.layers[1].midpoints, subclass_indices=self.layers[0].subclass_indices, 
            subclass_points=subclass_points, subclass_labels=subclass_labels, parameters=self.parameters, support_vectors=support_vectors,
            sum_dual_coef_diff=sum_dual_coef_diff, sum_dual_coef_sub=sum_dual_coef_sub, diff_mult_factor=self.layers[0].mult_factor,
            sub_mult_factor=self.layers[1].mult_factor, label_lengths=label_lengths, symbolic=symbolic)
        print('Saved as ' + full_filename)
    
    def save_convolutional_network(self, filename):
        """Save the information from a convolutional Network so that it can be reloaded; each layer will be saved
        in a different file, with a single parameter file in addition"""
        num_points = 0
        for sc in self.subclasses:
            num_points += len(sc.points)
        subclass_points = np.zeros((num_points,))
        subclass_labels = np.zeros((len(self.subclasses), 2))
        for sc_i, sc in enumerate(self.subclasses):
            subclass_points[sc.points] = sc_i
            subclass_labels[sc_i] = sc.y_class
        
        np.savez(filename + 'p.npz', parameters=self.parameters, subclass_points=subclass_points, subclass_labels=subclass_labels)
        for l_i,layer in enumerate(self.layers):
            np.savez(filename + str(l_i) + '.npz', weights=layer.weights, biases=layer.biases, activation_function=layer.activation_function,
            subclass_indices=layer.subclass_indices, midpoints=layer.midpoints, sum_dual_coef=layer.sum_dual_coef,
            support_vectors=layer.support_vectors, mult_factor=layer.mult_factor, symbolic=layer.symbolic, multilabel=layer.multilabel, label_lengths=layer.label_lengths,
            convolutional=layer.convolutional, flatten_output=layer.flatten_output, win_size=layer.win_size, pooling=layer.pooling,
            stride=layer.stride, padding=layer.padding)

    def load_network(self, filename):
        """Loads a network"""
        
        if filename[-5]=='p':
            self.load_conv_network(filename)
            return

        network_file = np.load(filename, allow_pickle=True)
        self.parameters = network_file['parameters'][()]

        num_subclasses = len(np.unique(network_file['subclass_points']))
        for sc in range(num_subclasses):
            new_points = [sc_i[0] for sc_i in enumerate(network_file['subclass_points']) if sc_i[1]==sc]
            new_subclass = Subclass(network_file['subclass_labels'][sc], new_points)
            self.subclasses.append(new_subclass)
        activations = network_file['activations']
        
        differentiae = Layer(activations[0])
        diff_weights = network_file['differentia_layer']
        differentiae.weights = diff_weights[:-1, :]
        differentiae.biases = diff_weights[-1, :]
        differentiae.midpoints = network_file['differentia_midpoints']
        differentiae.subclass_indices = network_file['subclass_indices']
        differentiae.support_vectors = []
        differentiae.sum_dual_coef = network_file['sum_dual_coef_diff']
        differentiae.mult_factor = float(network_file['diff_mult_factor'])
        for i in range(differentiae.num_nodes()):
            differentiae.support_vectors.append(network_file['support_vectors'][np.where(network_file['support_vectors'][:, 0]==i), 1][0].astype(int))
        if 'symbolic' in network_file:
            differentiae.symbolic = bool(network_file['symbolic'][0])
        self.add_layer(differentiae)

        subclass_layer = Layer(activations[1])
        subclass_weights = network_file['subclass_layer']
        subclass_layer.weights = subclass_weights[:-1, :]
        subclass_layer.biases = subclass_weights[-1, :]
        subclass_layer.midpoints = network_file['subclass_midpoints']
        subclass_layer.sum_dual_coef = network_file['sum_dual_coef_sub']
        subclass_layer.mult_factor = float(network_file['sub_mult_factor'])
        subclass_layer.levels = np.zeros(subclass_layer.num_nodes())
        if 'symbolic' in network_file:
            differentiae.symbolic = bool(network_file['symbolic'][1])
        self.add_layer(subclass_layer)

        concept_layer = Layer(activations[2])
        concept_weights = network_file['concept_layer']
        concept_layer.weights = concept_weights[:-1, :]
        concept_layer.biases = concept_weights[-1, :]
        if 'label_lengths' in network_file:
            if network_file['label_lengths'].shape:
                concept_layer.multilabel = True
                concept_layer.label_lengths = network_file['label_lengths']
        if 'symbolic' in network_file:
            differentiae.symbolic = bool(network_file['symbolic'][2])

        self.add_layer(concept_layer)
    
    def load_sgdnet(self, filename):
        """Loads an SGD network, which does not have all of the same meta-data"""

        if filename[-5]=='p':
            self.load_conv_network(filename)
            return

        network_file = np.load(filename, allow_pickle=True)
        self.parameters = network_file['parameters'][()]
        activations = network_file['activations']
        
        layer0 = Layer(activations[0])
        lay0_weights = network_file['layer0']
        layer0.weights = lay0_weights[:-1, :]
        layer0.biases = lay0_weights[-1, :]
        self.add_layer(layer0)

        layer1 = Layer(activations[1])
        lay1_weights = network_file['layer1']
        layer1.weights = lay1_weights[:-1, :]
        layer1.biases = lay1_weights[-1, :]
        self.add_layer(layer1)

        layer2 = Layer(activations[2])
        lay2_weights = network_file['layer2']
        layer2.weights = lay2_weights[:-1, :]
        layer2.biases = lay2_weights[-1, :]
        self.add_layer(layer2)
    
    def load_conv_network(self, filename):
        """Loads a convolutional neural network from a saved file"""

        if os.path.isfile(filename): #sgd nets don't have parameter files, so don't try
            network_file = np.load(filename, allow_pickle=True)
            self.parameters = network_file['parameters'][()]

            num_subclasses = len(np.unique(network_file['subclass_points']))
            for sc in range(num_subclasses):
                new_points = [sc_i[0] for sc_i in enumerate(network_file['subclass_points']) if sc_i[1]==sc]
                new_subclass = Subclass(network_file['subclass_labels'][sc], new_points)
                self.subclasses.append(new_subclass)
        
        for l_i in range(100):
            filename = list(filename)
            filename[-5] = str(l_i)
            filename = ''.join(filename)
            if not os.path.isfile(filename):
                break
            network_file = np.load(filename, allow_pickle=True)
            new_layer = Layer()
            new_layer.activation_function = str(network_file['activation_function'])
            new_layer.weights = network_file['weights']
            new_layer.biases = network_file['biases']
            new_layer.subclass_indices = network_file['subclass_indices']
            new_layer.midpoints = network_file['midpoints']
            new_layer.sum_dual_coef = network_file['sum_dual_coef']
            new_layer.support_vectors = network_file['support_vectors']
            new_layer.mult_factor = float(network_file['mult_factor'])
            new_layer.multilabel = bool(network_file['multilabel'])
            new_layer.label_lengths = network_file['label_lengths']
            if 'symbolic' in network_file:
                new_layer.symbolic = bool(network_file['symbolic'])
            
            new_layer.convolutional = bool(network_file['convolutional'])
            if new_layer.convolutional:
                new_layer.win_size = int(network_file['win_size'])
                new_layer.pooling = int(network_file['pooling'])
                new_layer.flatten_output = bool(network_file['flatten_output'])
                new_layer.stride = int(network_file['stride'])
                if 'padding' in network_file:
                    new_layer.padding = bool(network_file['padding'])
            self.add_layer(new_layer)

    def SaveNetwork(self, fileName):
        """Saves the network"""
        fileNameNpz = fileName + '.npz'

        #Get all of the necessary information and then store
        hL1 = np.concatenate((self.layers[0].weights, self.layers[0].biases.reshape(1, -1)))
        hL2 = np.concatenate((self.layers[1].weights, self.layers[1].biases.reshape(1, -1)))
        hL3 = np.concatenate((self.layers[2].weights, self.layers[2].biases.reshape(1, -1)))
        activations = [l.activation_function for i,l in enumerate(self.layers)]
        sumDualCoefDiff = self.layers[0].sum_dual_coef
        sumDualCoefSub = self.layers[1].sum_dual_coef
        
        nPoints = 0
        for sc in self.subclasses:
            nPoints += len(sc.points)

        if self.multilabel:
            subclass_points = np.zeros((nPoints,))
            subclass_labels = np.zeros((len(self.subclasses), self.layers[2].num_nodes()))
            for sc_i, sc in enumerate(self.subclasses):
                subclass_points[sc.points] = sc_i
                subclass_labels[sc_i] = sc.y_class[0]
        else:
            subclass_points = np.zeros((nPoints,))
            subclass_labels = np.zeros((len(self.subclasses), 2))
            for sc_i, sc in enumerate(self.subclasses):
                subclass_points[sc.points] = sc_i
                subclass_labels[sc_i] = sc.y_class
        
        if self.layers[2].multilabel:
            labelLengths = self.layers[2].label_lengths
        else:
            labelLengths = self.layers[2].num_nodes()
        
        symbolic = [l.symbolic for l in self.layers]
        
        supportVectors = np.zeros((0,2))
        for i, svs in enumerate(self.layers[0].support_vectors):
            diff_svs = np.hstack((np.ones((len(svs), 1))*i, svs.reshape(-1, 1)))
            supportVectors = np.vstack((supportVectors, diff_svs))

        np.savez(fileNameNpz, differentia_layer=hL1, subclass_layer=hL2, concept_layer=hL3, activations=activations,
            differentia_midpoints=self.layers[0].midpoints, subclass_midpoints=self.layers[1].midpoints, subclass_indices=self.layers[0].subclass_indices, 
            subclass_points=subclass_points, subclass_labels=subclass_labels, parameters=self.parameters, support_vectors=supportVectors,
            sum_dual_coef_diff=sumDualCoefDiff, sum_dual_coef_sub=sumDualCoefSub, diff_mult_factor=self.layers[0].mult_factor,
            sub_mult_factor=self.layers[1].mult_factor, label_lengths=labelLengths, symbolic=symbolic)
        print('Saved as ' + fileNameNpz)
    