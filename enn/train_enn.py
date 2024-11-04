"""Trains an ENN"""

#Imports
import numpy as np
import scipy.sparse as sparse
import math
import scipy.cluster
import scipy.sparse as sparse
from enn.subclass import Subclass
from enn.network import Network
from enn.layer import Layer
from enn.enn_svm import SVM
from scipy import stats
import copy
from sklearn.cluster import KMeans
import random
import os
import time
from multiprocessing import Pool
import multiprocessing
from functools import partial
import scipy.sparse as sparse
from sklearn.linear_model import LogisticRegression

def train_full_enn(x_train, y_train, parameters, x_val=None, y_val=None, x_train_0=None):
    """Trains an ENN with training data, training labels, and a set of parameters"""

    def train():
        """The main training function"""
        global subclasses, differentiae, subclass_layer, standardization, convolutional_layers
        nonlocal x_train, x_train_0
        
        time_start = time.process_time()
        wall_time_start = time.time()
        print('Start:',time.time())

        #Make sure all necessary parameters are given, otherwise set to defaults
        if not check_parameters():
            return None

        #Perform cross_validation if asked for; if no cross-validation, otherwise print parameters
        if parameters['cross_val_fold']>1: 
            cross_validate(parameters['cross_val_fold'])
            return None
        elif parameters['cross_val_fold']==0:
            print('Parameters', end=': ')
            print(parameters)
            print()
        
        #An option to standardize the mean and variance of each input
        if parameters['standardize_inputs']:
            standardization = standardize()

        #If the ENN will have convolutional layers, train these layers first
        convolutional_layers = []
        if parameters['convolutional']:
            x_train_0 = x_train.copy()
            for conv_layer in range(len(parameters['convolution_num'])):
                #Train a convolutional layer with given parameters
                convolutional_layers.append(get_convolutional(parameters['convolution_num'][conv_layer], parameters['convolution_size'][conv_layer],
                    parameters['convolution_pooling'][conv_layer], parameters['convolution_padding'][conv_layer], parameters['convolution_stride'][conv_layer], parameters['convolution_strict'][conv_layer], improve=(random.random()<.5 and conv_layer==len(parameters['convolution_num'])-1)))
                if conv_layer<len(parameters['convolution_num'])-1:
                    convolutional_layers[-1].flatten_output = False
                convolutional_layers[-1].set_mult_factor(parameters['convolution_mult'][conv_layer])

                #Replace the training data with the output of the convolutional layer
                x_train = convolutional_layers[-1].compute_output(x_train)
        
        
        #Train the ENN on the data (or on the convolutional output)
        if not parameters['grid_search']:
            #These will find the subclasses and then find the differentia, subclass, and concept nodes
            subclasses = get_subclasses(parameters['num_subclasses'])
            differentiae = get_all_differentiae(parameters['svm_cost_1'], parameters['d_strictness'])
            subclass_layer = get_subclass_neurons(parameters['svm_multiplier_1'], parameters['svm_cost_2'],
                parameters['margin_ratio'], parameters['misclass_tolerance'], parameters['sc_strictness'])
            concept_layer = get_concept_layer(parameters['svm_multiplier_2'], parameters['c_strictness'])
            
            #Build the network from the layers
            if parameters['standardize_inputs']:
                destandardize() #changes the network parameters to handle non-standardized inputs
            network = Network(parameters)
            network.subclasses = subclasses
            for layer in convolutional_layers:
                network.add_layer(layer)
            differentiae.desparsify()
            subclass_layer.desparsify()
            concept_layer.desparsify()
            network.add_layer(differentiae)
            network.add_layer(subclass_layer)
            network.add_layer(concept_layer)

            """
            for l,layer in enumerate(network.layers):
                if layer.activation_function == 'tanh':
                    layer.activation_function = 'sigmoid'
                    if l<len(network.layers)-1:
                        network.layers[l+1].tanh_to_sigmoid()
            """
            #If performing cross-validation, compute the validation error
            if x_val is not None:
                print('(', differentiae.num_nodes(), subclass_layer.num_nodes(), ')', end=' ')
                network.compute_error(x_val, y_val)
        else:
            #The following does a grid search through relevant parameters and computes the validation error
            subclasses = get_subclasses(parameters['num_subclasses'])
            for svm_cost_1 in parameters['svm_cost_1']:
                differentiae = get_all_differentiae(svm_cost_1)
                original_differentiae = differentiae.copy()
                for svm_multiplier_1 in parameters['svm_multiplier_1']:
                    for svm_cost_2 in parameters['svm_cost_2']:
                        for misclass_tolerance in parameters['misclass_tolerance']:
                            differentiae = original_differentiae.copy()
                            for sc in subclasses:
                                sc.first_margin = None
                                sc.first_misclass = None
                            for margin_ratio in parameters['margin_ratio']:    
                                subclass_layer = get_subclass_neurons(svm_multiplier_1, svm_cost_2, margin_ratio, misclass_tolerance)
                                for svm_multiplier_2 in parameters['svm_multiplier_2']:
                                    concept_layer = get_concept_layer(svm_multiplier_2)
                                    
                                    #Build ENN
                                    new_parameters = copy.deepcopy(parameters)
                                    new_parameters['svm_cost_1'] = svm_cost_1
                                    new_parameters['svm_multiplier_1'] = svm_multiplier_1
                                    new_parameters['svm_cost_2'] = svm_cost_2
                                    new_parameters['margin_ratio'] = margin_ratio
                                    new_parameters['misclass_tolerance'] = misclass_tolerance
                                    new_parameters['svm_multiplier_2'] = svm_multiplier_2
                                    network = Network(new_parameters)
                                    network.subclasses = subclasses
                                    for layer in convolutional_layers:
                                        network.add_layer(layer)
                                    network.add_layer(differentiae)
                                    network.add_layer(subclass_layer)
                                    network.add_layer(concept_layer)

                                    #Compute validation error and print information
                                    err = network.compute_error(x_val, y_val, print_error=False)
                                    print(new_parameters)
                                    print('Validation Time (s):', time.process_time() - time_start)
                                    print(differentiae.num_nodes(), subclass_layer.num_nodes(), end=' ')
                                    print('Error: ', round(err[0], 3))

        #Find and print the CPU and wall time of running
        wall_time_stop = time.time()
        time_stop = time.process_time()
        print('Process Time (s):', time_stop-time_start)
        print('Wall Time (s):', wall_time_stop-wall_time_start)
        
        #End
        return network

    def cross_validate(cross_val_fold):
        """Split the data up and train multiple ENNs to get cross-validation error"""
        parameters['cross_val_fold'] = -1
        rand_order = np.argsort(np.random.rand(y_train.shape[0]))
        num_test = y_train.shape[0]/cross_val_fold
        start_ind = 0
        end_ind = num_test
        for trial in range(cross_val_fold):
            print('Cross Validation Trial ', trial+1)
            test_ind = rand_order[int(start_ind):int(end_ind)]
            train_ind = np.setdiff1d(range(y_train.shape[0]), test_ind)
            train_full_enn(x_train[train_ind, :], y_train[train_ind], parameters, x_val=x_train[test_ind, :], y_val=y_train[test_ind])
            start_ind += num_test
            end_ind += num_test
    
    def standardize():
        """Standard each input feature to mean 0 and standard deviation 1"""
        nonlocal x_train
        standardization = np.zeros((x_train.shape[1], 2))
        standardization[:, 0] = np.mean(x_train, axis=0)
        standardization[:, 1] =  np.std(x_train, axis=0)
        x_train -= np.tile(np.reshape(standardization[:, 0], (1, -1)), (x_train.shape[0], 1))
        x_train = np.divide(x_train, np.tile(np.reshape(standardization[:, 1], (1, -1)), (x_train.shape[0], 1)))
        return standardization
    
    def destandardize():
        """Changes the parameters in the first layer so that the network can run on non-standardized inputs"""
        nonlocal x_train
        for dim in range(x_train.shape[1]):
            differentiae.biases -= np.divide(np.multiply(differentiae.weights[dim, :], standardization[dim, 0]), standardization[dim, 1])
        for dim in range(x_train.shape[1]):
            differentiae.weights[dim, :] /= standardization[dim, 1]
        x_train = np.multiply(x_train, np.tile(np.reshape(standardization[:, 1], (1, -1)), (x_train.shape[0], 1)))
        x_train += np.tile(np.reshape(standardization[:, 0], (1, -1)), (x_train.shape[0], 1))

    def get_convolutional(num_filters, win_size=3, pooling=2, padding=False, stride=1, strictness=0, improve=False):
        """Returns a convolutional layer"""
        if 'SLURM_JOB_ID' in os.environ:
            window_samples = 800000 #an arbitrary number chosen because it allows for reasonable run time
            if len(x_train)<=10000:
                window_samples = 20000
        else:
            window_samples = 20000           
        
        #Sets up things to collect sample windows from across all images
        numel = x_train.shape[1]
        depth = 1
        if len(x_train.shape)>2:
            depth = x_train.shape[2]
        window_indices = Layer.get_subimage_indices(x_train.shape[1:], win_size)
        num_windows = len(window_indices)
        window_samples_per_class = math.ceil(window_samples/num_windows/len(parameters['classes']))
        window_samples = window_samples_per_class*num_windows*len(parameters['classes'])
        labels = np.zeros(window_samples)

        samples = np.zeros((window_samples, (win_size**2)*depth))
        class_ind = []
        for c in range(len(parameters['classes'])):
            class_ind.append(np.array([i for i,y in enumerate(y_train) if y==parameters['classes'][c]]))
        cur_sample_ind = 0

        #Moving across the image, it selects small sub-images at random the size of the convolutional filter

        for indices in window_indices:
            for c in range(len(parameters['classes'])):
                num_samples = min(window_samples_per_class, len(class_ind[c]))
                sample_ind = class_ind[c][random.sample(range(len(class_ind[c])), num_samples)]
                if depth==1:
                    window_sample = x_train[sample_ind, :][:, indices[0]]
                else:
                    window_sample = x_train[sample_ind, :, :][:, indices[0], :]
                    window_sample = window_sample.reshape((num_samples,-1))
                samples[cur_sample_ind:cur_sample_ind+num_samples, :] = window_sample
                labels[cur_sample_ind:cur_sample_ind+num_samples] = c 
                cur_sample_ind += num_samples
        if cur_sample_ind<window_samples:
            samples = samples[:cur_sample_ind, :]
            labels = labels[:cur_sample_ind]
        print('Convolutional layer window samples:', samples.shape[0])

        #Cluster the window images into different "subconcepts"
        #improve = False
        if improve:
            print('Improving convolutional filters')
            clustering = KMeans(n_clusters=num_filters*2, verbose=False, n_init=2).fit(samples)
            entropy = np.zeros(num_filters*2)
            all_labels = np.unique(labels)
            for i in range(num_filters*2):
                n = np.where(clustering.labels_==i)[0]
                p = [np.sum(labels[n]==c)/len(n) for c in range(len(parameters['classes']))]
                entropy[i] = -sum([p_i*math.log(p_i) for p_i in p if p_i>0])
            #cluster_centers = improve_convolutional_centers(clustering, win_size, pooling, padding, stride)
            cluster_centers = clustering.cluster_centers_[np.argsort(entropy)[:num_filters]]
        else:
            clustering = KMeans(n_clusters=num_filters, verbose=False, n_init=1).fit(samples)
            cluster_centers = clustering.cluster_centers_

        #Build the convolutional layer by separating each cluster center from all the others
        convolutional_layer = Layer('tanh', convolutional=True, convolution_win_size=win_size, pooling=pooling, padding=padding, stride=stride)     
        random.seed(int(parameters['job_id']))
        if True:#random.random()<1/3:
            print('Convolutional filters: basic method')
            for filter in range(num_filters):
                svm_labels = -np.ones(num_filters)
                svm_labels[filter] = 1
                svm = SVM(cluster_centers, svm_labels, svm_cost=1, fast=parameters['fast_svm'])
                svm.train()
                convolutional_layer.add_node(svm, subclass_1=filter)
        elif random.random()<1/2:
            print('Convolutional filters as differentiae')
            
            
            indices = []
            count = 0
            for c0 in range(num_filters):
                indices.extend([count+i for i in range(c0+1,num_filters)])
                count = indices[-1]+1
            
            separation_factor = 2
            num_trials = 5
            for trial in range(num_trials):
                all_svms = []
                num_svm = int(num_filters*(num_filters-1)/2)
                separations = np.zeros((num_svm, len(indices)))
                
                ys = np.array([1,-1])
                count = 0
                for c0 in range(num_filters):
                    for c1 in range(c0+1,num_filters):
                        svm = SVM(cluster_centers[[c0,c1],:], ys, svm_cost=1, fast=parameters['fast_svm'])
                        svm.train()
                        svm.trim()
                        all_svms.append(svm)
                        distances = (cluster_centers @ svm.weights) + svm.bias
                        distances_x = np.tile(distances.reshape((-1,1)), (1, num_filters))
                        distances_y = np.tile(distances.reshape((1,-1)), (num_filters, 1))

                        diff_signs = 1-np.equal(np.sign(distances_x), np.sign(distances_y))
                        ratios = np.abs(distances_x/distances_y)
                        far_apart = np.minimum(ratios<separation_factor, ratios>1/separation_factor)
                        separations[count] = np.minimum(diff_signs, far_apart).flatten()[indices]
                        count += 1
                
                total_separations = np.zeros((1,len(indices)))
                svms_to_keep = []
                for f in range(num_filters):
                    possible_separations = separations + np.tile(total_separations, (len(separations), 1))
                    overall_min = np.min(possible_separations)
                    possible_total_separations = np.sum(possible_separations<=overall_min, axis=1)
                    svm_ind = np.argmin(possible_total_separations)
                    total_separations[0] += separations[svm_ind]
                    svms_to_keep.append(all_svms.pop(svm_ind))
                    separations = np.delete(separations, svm_ind, axis=0)
                
                if np.min(total_separations)>0 or trial>=num_trials-1:
                    for filter in range(num_filters):
                        convolutional_layer.add_node(svms_to_keep[filter], subclass_1=filter)
                    break
                separation_factor *= 1.5


        elif True:
            print('Convolutional filters iteratively')
            cluster_centers = cluster_centers.copy()
            original_ids = [i for i,_ in enumerate(cluster_centers)]

            for t in range(num_filters):
                svms = []
                errors = []
                temp_num_filters = len(cluster_centers)
                if temp_num_filters==0:
                    break
                for filter in range(temp_num_filters):
                    svm_labels = -np.ones(temp_num_filters)
                    svm_labels[filter] = 1
                    svm = SVM(cluster_centers, svm_labels, svm_cost=1, fast=parameters['fast_svm'])
                    svm.train()
                    svms.append(svm)
                    errors.append(svm.get_misclass_error())
                if t<num_filters:
                    #min_err = min(errors)
                    filters_to_add = np.argsort(errors)
                    filters_to_add = filters_to_add[:min(temp_num_filters,int(round(num_filters/10)))].tolist()
                    #filters_to_add = [i for i,e in enumerate(errors) if e<=min_err*1.00001]
                    if temp_num_filters-len(filters_to_add)==1:
                        del filters_to_add[-1]
                    #if min_err>0:
                    #    filters_to_add = [filters_to_add[0]]
                else:
                    filters_to_add = [i for i in range(temp_num_filters)]
                for filter in filters_to_add:
                    convolutional_layer.add_node(svms[filter], subclass_1=original_ids[filter])
                for filter in reversed(np.sort(filters_to_add)):
                    del original_ids[filter]
                cluster_centers = np.delete(cluster_centers, filters_to_add, axis=0)

        return convolutional_layer
    
    def improve_convolutional_centers(clustering, win_size, pooling, padding, stride):
        centers = clustering.cluster_centers_
        num_filters_final = int(len(centers)/2)
        max_min_margin = -np.inf
        num_subc = min(50, parameters['num_subclasses'])
        temp_centers = centers.copy()
        while len(temp_centers)>num_filters_final:
            
            num_filters = len(temp_centers)
            convolutional_layer = Layer('tanh', convolutional=True, convolution_win_size=win_size, pooling=pooling, padding=padding, stride=stride)
            convolutional_layer.flatten_output = False
            for filter in range(num_filters):
                svm_labels = -np.ones(num_filters)
                svm_labels[filter] = 1
                svm = SVM(temp_centers, svm_labels, svm_cost=1, fast=parameters['fast_svm'])
                svm.train()
                convolutional_layer.add_node(svm, subclass_1=filter)
            
            inds_to_use = range(len(x_train))
            if len(x_train)>2000:
                inds_to_use = random.sample(inds_to_use, 2000)
            output = convolutional_layer.compute_output(x_train[inds_to_use])

            subc = get_subclasses(num_subc, x=output.reshape((len(inds_to_use), -1)), y=y_train[inds_to_use], hierarchical=0, suppl_unsupervised=0)

            total_error = 0
            min_margin = 0#np.inf
            count = 0
            total_h = None
            for i,s0 in enumerate(subc):
                for j,s1 in enumerate(subc):
                    if i<=j:
                        continue
                    if np.all(s0.y_class == s1.y_class):
                        continue
                    mn0 = np.mean(output[s0.points],axis=0)
                    mn1 = np.mean(output[s1.points],axis=0)
                    h = (mn0-mn1)
                    
                    b = -(.5*(mn0+mn1).reshape(1,-1) @ h.reshape(-1, 1))
                    

                    #d0 = output[s0.points].reshape(len(s0.points), -1) @ h.reshape(-1, 1) + b
                    #d1 = output[s1.points].reshape(len(s1.points), -1) @ h.reshape(-1, 1) + b
                    #total_error += (np.mean(d0<0) + np.mean(d1>0))/2
                    #min_margin = min(min_margin, (min(d0)-max(d1))[0])
                    #min_margin += min(0,(min(d0)-max(d1))[0])
                    count += 1
                    if total_h is None:
                        total_h = np.sum(h**2, axis=0)
                    else:
                        total_h += np.sum(h**2, axis=0)
            
            total_error /= count
            min_margin /= count
            print('Error:',total_error,'Margin:',round(min_margin,3))
            
            ind = np.argmin(total_h)
            temp_centers = np.delete(temp_centers, ind, axis=0)

            """
            if total_error < best_error:
                best_error = total_error
                centers = temp_centers
                sd = .005
                num_centers_to_change = 4
                max_min_margin = min_margin
                print('    Found new best error of',total_error)
            elif total_error <= best_error*1.001 and min_margin > max_min_margin:
                max_min_margin = min_margin
                centers = temp_centers
                print('    Found new best margin of',min_margin)
            else:
                sd *= 1.05
                if sd>.02:
                    sd = .02
                    num_centers_to_change *= 1.05
                    num_centers_to_change = min(num_centers_to_change, num_filters)
            """

            """
            distances = np.zeros(num_samples)
            labels = np.zeros(num_samples)
            for sample in range(num_samples):
                inds = np.random.choice(range(len(output)), size=2, replace=False)
                labels[sample] = int(y_train[inds_to_use][inds[0]]==y_train[inds_to_use][inds[1]])
                distances[sample] = np.sqrt(np.sum((output[inds[0]]-output[inds[1]])**2))
            
            clf = LogisticRegression().fit(distances.reshape(-1,1), labels)
            
            probs = [.95, .05]
            ds = [0, 0]
            for i,p in enumerate(probs):
                d_hi = np.max(distances)
                d_lo = np.min(distances)
                for _ in range(20):
                    trial_dist = np.array([d_lo, d_hi])
                    temp_output = clf.predict_proba(trial_dist.reshape(-1,1))
                    min_ind = np.argmin(np.abs(p-temp_output))
                    if min_ind == 0:
                        d_hi = d_lo*.4 + d_hi*.6
                    else:
                        d_lo = d_lo*.6 + d_hi*.4
                ds[i] = .5*(d_lo+d_hi)
            avg_spacing = ds[1]-ds[0]
            

            print('Found spacing of',avg_spacing)
            if avg_spacing>best_spacing:
                best_spacing = avg_spacing
                centers = temp_centers
                print('Improved convolutional filters to spacing of',avg_spacing)
            """
        
        return temp_centers


    def get_subclasses(num_subclasses, coarse_grained=0, x=None, y=None, hierarchical=None, suppl_unsupervised=None):
        """Divides up each class into subclasses"""
        classes = [] #A list of subclass objects representing each class
        linkage_trees = [] #Linkage trees for each class
        curr_ind = [] #The current branch point for each linkage tree
        cluster_ids = [] #The indices of all the split branches for each class
        
        if x is None or y is None:
            x = x_train
            y = y_train
            print('Getting subconcepts')
            print()
        if parameters['alternative_subclasses']==1:
            return get_subclasses_alternative_old(num_subclasses)
        if hierarchical is None:
            hierarchical = parameters['hierarchical_subclasses']
        if suppl_unsupervised is None:
            suppl_unsupervised = parameters['suppl_unsupervised_subconcepts']

        if coarse_grained:
            num_subclasses = coarse_grained

        #First we will divide up all the points into their various classes and compute a linkage tree for each
        if num_subclasses>0 and (num_subclasses<x.shape[0] or hierarchical):
            use_hierarchical = True
            if use_hierarchical:
                cl_metric = 'ward'
                if parameters['regression']:
                    bin_values = np.sort(y)[np.round(np.linspace(0, y.shape[0]-1, parameters['regression']+1)).astype(int)]
                    bin_values[-1] = np.inf
                    bin_id = np.digitize(y, bin_values)-1
                
                #Perform clustering on each class
                for c in parameters['classes']:
                    new_subclass = Subclass(y_class=[c, 0])
                    if parameters['regression']:
                        new_subclass.reset_points([i for i, yi in enumerate(bin_id) if yi==c])
                    elif parameters['multilabel']:
                        new_subclass.reset_points([i for i, yi in enumerate(y) if np.all(yi==c)])
                    else:
                        new_subclass.reset_points([i for i, yi in enumerate(y) if yi==c])
                    classes.append(new_subclass)
                    if len(new_subclass.points)>1:
                        #Get the linkage tree
                        if parameters['clustering_weights'] is not None:
                            #This is an option to weight certain input values when performing the clustering
                            data = x[new_subclass.points]
                            if len(parameters['clustering_weights'].shape)>1:
                                data[:, parameters['clustering_weights'][:,0].astype(int)] *= parameters['clustering_weights'][0,1]
                            else:
                                data = data * np.tile(parameters['clustering_weights'].reshape(1,-1), (len(data), 1))
                            if sparse.issparse(data):
                                if data.shape[0]*data.shape[1] < 1e7:
                                    linkage_trees.append(scipy.cluster.hierarchy.linkage(data.toarray(), method=cl_metric))
                                else:
                                    linkage_trees.append(scipy.cluster.hierarchy.linkage(data[:,parameters['clustering_weights'][:,0].astype(int)].todense(), method='single'))
                            else:
                                linkage_trees.append(scipy.cluster.hierarchy.linkage(data, method=cl_metric))
                        elif coarse_grained:
                            if x.shape[1]==784:
                                dim = 5
                                indices = np.reshape(np.arange(784), (28, 28))
                                data = np.zeros((len(new_subclass.points), dim**2))
                                count = 0
                                for i in range(dim):
                                    start_i = int(math.floor(28/dim*i))
                                    stop_i = int(math.ceil(28/dim*(i+1)))
                                    for j in range(dim):
                                        start_j = int(math.floor(28/dim*j))
                                        stop_j = int(math.ceil(28/dim*(j+1)))
                                        sub_image = x[new_subclass.points, :][:, indices[start_i:stop_i,:][:,start_j:stop_j].flatten()]
                                        data[:, count] = np.mean(sub_image, axis=1)
                                        count += 1
                                linkage_trees.append(scipy.cluster.hierarchy.linkage(data, method=cl_metric))
                            else:
                                return []
                        else:
                            #The standard way to get a linkage tree
                            data = x[new_subclass.points]#/np.sum(x[new_subclass.points], axis=0)
                            linkage_trees.append(scipy.cluster.hierarchy.linkage(data, method=cl_metric))
                            #linkage_trees.append(scipy.cluster.hierarchy.linkage(data, method=cl_metric))
                    else:
                        linkage_trees.append(None)
                    curr_ind.append(len(new_subclass.points)-2)
                    cluster_ids.append([len(new_subclass.points)*2-2])

                #We split each of the classes into subclasses, finding which to split next by finding the largest next branch
                #First we find the order of splitting them
                max_num_subclasses = num_subclasses
                if parameters['alternative_subclasses']:
                    max_num_subclasses = 30
                for num_subs in range(num_subclasses, max_num_subclasses+1):
                    if num_subs>num_subclasses:
                        curr_ind = []
                        cluster_ids = []
                        for c in range(len(classes)):
                            curr_ind.append(len(classes[c].points)-2)
                            cluster_ids.append([len(classes[c].points)*2-2])
                    for c in range(len(classes), num_subs):
                        max_branch = -1
                        to_split = 0
                        for i in range(len(classes)):
                            if curr_ind[i]<0:
                                continue
                            if linkage_trees[i] is None:
                                continue
                            if linkage_trees[i][curr_ind[i]][2] > max_branch:
                                max_branch = linkage_trees[i][curr_ind[i]][2]
                                to_split = i
                            elif linkage_trees[i][curr_ind[i]][2] == max_branch:
                                if len(cluster_ids[i]) < len(cluster_ids[to_split]):
                                    max_branch = linkage_trees[i][curr_ind[i]][2]
                                    to_split = i
                        new_inds = linkage_trees[to_split][curr_ind[to_split]][0:2]
                        old_ind = cluster_ids[to_split].index(curr_ind[to_split] + len(classes[to_split].points))
                        cluster_ids[to_split].pop(old_ind)
                        cluster_ids[to_split].append(int(new_inds[0]))
                        cluster_ids[to_split].append(int(new_inds[1]))
                        curr_ind[to_split] -= 1
                    
                    max_branch = -1
                    for i in range(len(classes)):
                        if linkage_trees[i] is None:
                            continue
                        if linkage_trees[i][curr_ind[i]][2] > max_branch:
                            max_branch = linkage_trees[i][curr_ind[i]][2]
                    
                    #Now that all subclass clusters have been identified, place them into subclass list
                    subclasses = []
                    for c_i,c in enumerate(classes):
                        if linkage_trees[c_i] is None:
                            subclasses.append(c)
                            continue
                        if not hierarchical:
                            range_c = range(1)
                        else:                
                            num_levels = len(cluster_ids[c_i])
                            num_levels = max(1,num_levels-(hierarchical-1))
                            range_c = range(num_levels)
                        for r in range_c:
                            if r>0 and len(cluster_ids[c_i])-r<1:
                                continue
                            clusters = scipy.cluster.hierarchy.cut_tree(linkage_trees[c_i], n_clusters=len(cluster_ids[c_i])-r)
                            if r>0:
                                new_ind = 0
                                for i in range(len(cluster_ids[c_i])-r):
                                    old_ind = np.where(clusters==i)[0]
                                    if np.any(old_clusters[old_ind]!=old_clusters[old_ind[0]]):
                                        new_level = int(np.max(levels[old_ind])+1)
                                        new_ind = i
                                        break
                            else:
                                new_level = 0
                            num_clusters = len(np.unique(clusters))
                            for s in range(num_clusters):
                                if r>0:
                                    #if max(range_c)-r >= parameters['hierarchical_subclasses']+1:
                                    #    continue
                                    if s!=new_ind:
                                        continue
                                cluster_indices = np.where(clusters==s)[0]
                                subclasses.append(Subclass(c.y_class, [c.points[i] for i in cluster_indices], level=r))
                            if len(range_c)>1:
                                old_clusters=clusters
                                if r==0:
                                    levels = np.zeros(len(clusters))
                                else:
                                    levels[old_ind] = new_level
                    

                    if num_subs<max_num_subclasses:
                        good_subclasses = True
                        total_err = 0
                        for i in range(len(subclasses)):
                            for j in range(i+1,len(subclasses)):
                                if subclasses[i].y_class[0] == subclasses[j].y_class[0]:
                                    continue
                                err, _ = get_error_and_distance(subclasses[i].points, subclasses[j].points)
                                total_err += err
                                if total_err>.0002:
                                    good_subclasses = False
                                    break
                            if not good_subclasses:
                                break
                        if good_subclasses:
                            break
            else:
                max_num_subclasses = num_subclasses*2
                num_subs = num_subclasses
                while num_subs<=max_num_subclasses:
                    subclasses = []
                    if parameters['clustering_weights'] is None:
                        data = x_train
                    else:
                        data = x_train * parameters['clustering_weights'].reshape(1,-1)
                    data = np.hstack((data, 1e6*y_train.reshape(-1,1)))
                    clusters = KMeans(num_subs).fit(data)
                    for i in range(num_subs):
                        ind = np.where(clusters.labels_==i)[0]
                        y = scipy.stats.mode(y_train[ind])[0][0]

                        subclasses.append(Subclass([y,0], ind))
                    
                    if num_subs<max_num_subclasses:
                        good_subclasses = True
                        total_err = 0
                        for i in range(len(subclasses)):
                            for j in range(i+1,len(subclasses)):
                                if subclasses[i].y_class[0] == subclasses[j].y_class[0]:
                                    continue
                                err, _ = get_error_and_distance(subclasses[i].points, subclasses[j].points)
                                total_err += err
                                if total_err>.0002:
                                    good_subclasses = False
                                    break
                            if not good_subclasses:
                                break
                        if good_subclasses:
                            break
                    num_subs += 1
        else:
            subclasses = []
            for s in range(min(num_subclasses, len(y))):
                subclasses.append(Subclass([y[s], 0], [s]))

        if parameters['superclasses']:
            distances = np.full((len(classes), len(classes)), np.inf)
            if hierarchical != 1:
                for c in classes:
                    subclasses.append(c)
            cur_level = max([s.level for s in subclasses])+1
            while len(classes)>2:
                min_dist = [np.inf, [0,0]]
                for i in range(len(classes)):
                    for j in range(i+1,len(classes)):
                        if np.isinf(distances[i,j]):
                            ci = classes[i].points
                            cj = classes[j].points
                            distances[i,j] = 2*len(ci)*len(cj)/(len(ci)+len(cj))
                            distances[i,j] *= np.sum((np.mean(x_train[ci], axis=0)-np.mean(x_train[cj], axis=0))**2)
                        if distances[i,j]<min_dist[0]:
                            min_dist[0] = distances[i,j]
                            min_dist[1] = [i,j]
                
                new_points = classes[min_dist[1][0]].points
                new_points.extend(classes[min_dist[1][1]].points)
                new_class = Subclass([random.random(), 0], new_points, level=cur_level)
                subclasses.append(new_class)
                del classes[min_dist[1][1]]
                del classes[min_dist[1][0]]
                classes.append(new_class)
                distances = np.delete(distances, min_dist[1], axis=0)
                distances = np.delete(distances, min_dist[1], axis=1)
                distances = np.hstack((distances, np.full((distances.shape[0], 1), np.inf)))
                distances = np.vstack((distances, np.full((1, distances.shape[1]), np.inf)))
                cur_level += 1
                

        if suppl_unsupervised and coarse_grained==0:
            subclasses.extend(get_unsupervised_subconcepts(suppl_unsupervised))

        return subclasses
    
    def get_unsupervised_subconcepts(num_subclasses):
        print('Getting unsupervised subconcepts')
        print()

        if check_key(parameters, 'linkage_tree'):
            linkage_tree = parameters['linkage_tree']
        else:
            linkage_tree = scipy.cluster.hierarchy.linkage(x_train, method='ward')
        
        subclasses = []
        range_c = range(1)
        #if parameters['hierarchical_subclasses']:
        #    range_c = range(num_subclasses+1)
        for r in range_c:
            if r>0 and num_subclasses-r<2:
                continue
            clusters = scipy.cluster.hierarchy.cut_tree(linkage_tree, n_clusters=num_subclasses-r)
            if r>0:
                new_ind = 0
                for i in range(num_subclasses-r):
                    old_ind = np.where(clusters==i)[0]
                    if np.any(old_clusters[old_ind]!=old_clusters[old_ind[0]]):
                        new_level = int(np.max(levels[old_ind])+1)
                        new_ind = i
                        break
            else:
                new_level = 0
            num_clusters = len(np.unique(clusters))
            for s in range(num_clusters):
                if r>0 and s!=new_ind:
                    continue
                cluster_indices = np.where(clusters==s)[0]
                rnd = random.random()
                subclasses.append(Subclass([rnd, 0], cluster_indices, level=new_level))
            if len(range_c)>1:
                old_clusters=clusters
                if r==0:
                    levels = np.zeros(len(clusters))
                else:
                    levels[old_ind] = new_level
        
        return subclasses


    def get_all_differentiae(svm_cost, strictness=0, prune_convolution=False):
        """Calculates all differentiae between pairs of subclasses"""
        nonlocal x_train

        print('Start Differentiae:',time.time())
        differentiae = Layer(symbolic=parameters['symbolic']) #initialize the differentia layer
        differentiae.subclass_indices = np.zeros((len(subclasses), len(subclasses))) #initialize the index matrix
        
        print('Training differentiae')
        all_svms = []
        sc_indices = []

        #Go through all pairs of subclasses and compute an SVM that will serve as the differentia neuron
        if parameters['parallel']:
            if parameters['verbose']:
                #print(len(os.sched_getaffinity(0)))
                print(multiprocessing.cpu_count())
            if x_train.size>5000*64*120:
                x_train_shape = x_train.shape
                r = random.randint(0,100000)
                file_name = 'x_train_' + str(r) +'.npy'
                np.save(file_name, x_train.reshape((x_train.shape[0],-1)))
                pool = Pool (24)
                results = pool.map(partial(get_all_differentiae_test, x_train=(file_name,x_train_shape), y_train=y_train, subclasses=subclasses, svm_cost=svm_cost, parameters=parameters, margin_ratio=parameters['margin_ratio'], intraclass=parameters['intraclass_diff'], symbolic=parameters['symbolic'], strictness=strictness), range(len(subclasses)))
                pool.close()
                pool.join()
            else:
                pool = Pool(24)
                results = pool.map(partial(get_all_differentiae_test, x_train=x_train, y_train=y_train, subclasses=subclasses, svm_cost=svm_cost, parameters=parameters, margin_ratio=parameters['margin_ratio'], intraclass=parameters['intraclass_diff'], symbolic=parameters['symbolic'], strictness=strictness), range(len(subclasses)))
                pool.close()
                pool.join()
            for result in results:
                all_svms.extend(result[0])
                sc_indices.extend(result[1])
            if not np.isinf(parameters['margin_ratio']):
                already_filled = np.zeros((len(subclasses), len(subclasses)))
                to_keep = []
                for s in range(len(all_svms)):
                    if already_filled[sc_indices[s][0], sc_indices[s][1]]:
                        continue
                    elif already_filled[sc_indices[s][1], sc_indices[s][0]]:
                        continue
                    if sc_indices[s][1]<sc_indices[s][0]:
                        sc_indices[s] = [sc_indices[s][1], sc_indices[s][0]]
                        #all_svms[s] = all_svms[s].flip_direction()
                    already_filled[sc_indices[s][0], sc_indices[s][1]] = 1
                    to_keep.append(s)
                all_svms = [all_svms[s] for s in to_keep]
                sc_indices = [sc_indices[s] for s in to_keep]

                
        else:
            for s_i in range(len(subclasses)):
                if parameters['verbose']:
                    print()
                    print(s_i, end=' - ')
                for s_j in range(len(subclasses)):
                    if s_j==s_i:
                        continue
                    if s_j<s_i and strictness==0:
                        continue
                    if parameters['verbose']:
                        print(s_j, end=' ')
                    if (not parameters['intraclass_diff']) and np.all(subclasses[s_i].y_class[0]==subclasses[s_j].y_class[0]):
                        continue
                    num_overlapping = len(np.intersect1d(subclasses[s_i].points, subclasses[s_j].points))
                    frac_overlapping = num_overlapping/min(len(subclasses[s_i].points), len(subclasses[s_j].points))
                    if frac_overlapping>.1:
                        continue
                    svm = SVM(x_train[subclasses[s_i].points,:], x_train[subclasses[s_j].points,:], svm_cost, fast=parameters['fast_svm'], symbolic=parameters['symbolic'], strictness=strictness)
                    svm.train()
                    svm.trim()
                    all_svms.append(svm)
                    sc_indices.append([s_i, s_j])
        differentiae.add_nodes(all_svms, sc_indices) #adds the svm differentia to the full layer
        
        #If we want to keep meta data for each svm, format them for the neural layer
        if all_svms[0].keep_meta_data:
            for i in range(len(all_svms)):
                svs_ind = np.concatenate((subclasses[sc_indices[i][0]].points, subclasses[sc_indices[i][1]].points))
                differentiae.support_vectors.append(svs_ind[all_svms[i].support_vector_indices])
        if parameters['verbose']:
            print()
        
        #This is used if one chooses the option of pruning filters from the convolutional layer
        if prune_convolution and parameters['convolutional']:
            weights = differentiae.weights
            num_filters = convolutional_layers[-1].num_nodes()
            weights = np.divide(weights, np.tile(np.max(np.absolute(weights), axis=1), (weights.shape[1], 1)).transpose())
            mean_all_weights = np.mean(np.absolute(weights), axis=1)
            mean_weights = np.zeros(num_filters)
            for filter in range(num_filters):
                mean_weights[filter] = np.mean(mean_all_weights[range(filter, len(mean_all_weights), num_filters)])
            worst_filters = np.argsort(mean_weights)[:num_filters-parameters['convolution_num']]
            convolutional_layers[-1].del_node(worst_filters.tolist())
            x_nodes_to_delete = []
            for filter in worst_filters:
                x_nodes_to_delete = np.concatenate((x_nodes_to_delete, np.arange(filter, x_train.shape[1], num_filters)))
            x_train = np.delete(x_train, x_nodes_to_delete, axis=1)
            differentiae = get_all_differentiae(svm_cost, strictness=parameters['d_strictness'], prune_convolution=False)
        
        if parameters['symbolic']:
            #Check for duplicate differentiae
            """
            if not differentiae.issparse:
                w = np.vstack((differentiae.weights, differentiae.biases.reshape(1,-1)))
            else:
                w = sparse.vstack((differentiae.weights, differentiae.biases.reshape(1,-1)))
            product = np.abs(w.transpose() @ w)
            self_product = np.asarray(np.sum(w.multiply(w), axis=0)).flatten()
            for i in range(len(self_product)):
                product[:,i] /= self_product[i]
            
            are_same = np.abs(product-1)<1e-4
            is_duplicate = np.zeros(len(product))
            for i in range(len(product)):
                for j in range(i+1, len(product)):
                    if are_same[i,j] and not is_duplicate[j]:
                        is_duplicate[j] = True
                        ind = differentiae.get_diff_subclasses(j)
                        differentiae.subclass_indices[ind[0],ind[1]] = i+1
                        differentiae.subclass_indices[ind[1],ind[0]] = -(i+1)

            to_remove = [i for i,v in enumerate(is_duplicate) if v]
            differentiae.del_node(to_remove)
            """

        print('End differentiae:',time.time())
        return differentiae

    def get_subclass_neurons(svm_multiplier_1, svm_cost_2, margin_ratio, misclass_tolerance, strictness=0):
        """Generates neurons that combine a subclass together"""

        print('Start Subconcepts:',time.time())
        if parameters['hierarchical_subclasses']:
            margin_ratio = np.inf

        print()
        if np.isinf(margin_ratio) and parameters['full_prunes']<=0:
            print('Training subconcept neurons from {} differentiae'.format(differentiae.num_nodes()))
        else:
            print('Pruning {} differentia neurons by training subconcept neurons'.format(differentiae.num_nodes()))

        #Get things set up
        differentiae.set_mult_factor(svm_multiplier_1)
        subclass_layer = Layer(symbolic=parameters['symbolic'])
        num_diff = differentiae.num_nodes()
        diff_used = np.zeros((num_diff,)) # maintains a list of already used differentiae
        

        #Get all of the subclass neurons
        if parameters['parallel']:
            if x_train.size>5000*64*120 and not differentiae.issparse or True:
                r = random.randint(0,100000)
                if not parameters['direct_differentiae']:
                    file_name = 'diff_output_' + str(r) + '.npy'
                    x_train_shape = (r, (x_train.shape[0], differentiae.num_nodes()), None)
                    np.save(file_name, differentiae.compute_output(x_train))
                else:
                    x_train_shape = [r, (x_train.shape[0], None), differentiae.num_nodes(), []]
                    for i,s in enumerate(subclasses):
                        file_name = 'diff_output_' + str(r) + '_' + str(i) + '.npy'
                        nodes = differentiae.get_subclass_diff(i)
                        x_train_shape[3].append(nodes)
                        np.save(file_name, differentiae.compute_output(x_train, use_nodes=nodes))
                saved_x_train = True
            else:
                saved_x_train = False
            
                
            num_levels = max([sc.level for sc in subclasses])+1
            #print('Flat hierarchical')
            margin_ratio = np.inf
            for level in range(num_levels):
                print('Start Pool:',time.time())
                pool = Pool(24)
                if saved_x_train:
                    sc_svms = pool.map(partial(get_subclass_weights_test, x_train=x_train_shape, y_train=y_train, subclasses=subclasses, differentiae=None, svm_cost_2=svm_cost_2, margin_ratio=margin_ratio, misclass_tolerance=misclass_tolerance, symbolic=parameters['symbolic'], multilabel=parameters['multilabel'], strictness=parameters['sc_strictness'], level=level, hierarchical=parameters['hierarchical_subclasses'], fast_svm=parameters['fast_svm'], subclass_layer=subclass_layer), range(len(subclasses)))
                else:
                    sc_svms = pool.map(partial(get_subclass_weights_test, x_train=x_train, y_train=y_train, subclasses=subclasses, differentiae=differentiae, svm_cost_2=svm_cost_2, margin_ratio=margin_ratio, misclass_tolerance=misclass_tolerance, symbolic=parameters['symbolic'], multilabel=parameters['multilabel'], strictness=parameters['sc_strictness'], level=level, hierarchical=parameters['hierarchical_subclasses'], fast_svm=parameters['fast_svm'], subclass_layer=subclass_layer, parameters=parameters), range(len(subclasses)))
                print('End pool 1:',time.time())
                pool.close()
                pool.join()
                print('End pool:',time.time())
                svms = []
                for svm in sc_svms:
                    if svm is None:
                        continue
                    svms.append(svm[0])
                    if not parameters['hierarchical_subclasses']:
                        new_diff_used = np.nonzero(svm[0].weights)[0]
                        diff_used[new_diff_used] = True
                
                subclass_layer.add_nodes(svms, levels=level)
                if level+1<num_levels:
                    file_name = 'subc_output_' + str(r) + '.npy'
                    np.save(file_name, subclass_layer.compute_output(differentiae.compute_output(x_train)))
        else:
            num_levels = max([sc.level for sc in subclasses])+1
            for level in range(num_levels):
                sc_svms = []
                for sc_i in range(len(subclasses)):
                    if subclasses[sc_i].level==level:
                        results = get_subclass_weights_test(sc_i, x_train=x_train, y_train=y_train, subclasses=subclasses, differentiae=differentiae, svm_cost_2=svm_cost_2, margin_ratio=margin_ratio, misclass_tolerance=misclass_tolerance, symbolic=parameters['symbolic'], multilabel=parameters['multilabel'], strictness=parameters['sc_strictness'], level=level, hierarchical=parameters['hierarchical_subclasses'], fast_svm=parameters['fast_svm'], subclass_layer=subclass_layer, parameters=parameters)
                        #results = get_subclass_weights(sc_i, diff_used, svm_cost_2, margin_ratio, misclass_tolerance, strictness, level=level, subclass_layer=subclass_layer)
                        sc_svms.append(results[0])
                        if np.sum(results[1]) != 2:
                            debug=1
                        diff_used = np.maximum(diff_used, results[1])
                subclass_layer.add_nodes(sc_svms, levels=level)        

        #If there was pruning, that meant that weights were found ignoring off-class differentiae, but we would
        #like to retrain the whole thing with all differentiae available to the network
        if not np.isinf(margin_ratio) or parameters['full_prunes']>0:
            parameters['full_prunes'] -= 1
            to_remove = [i for i,d in enumerate(diff_used) if not d]
            differentiae.del_node(to_remove)
            if parameters['direct_differentiae']:
                subclass_layer.del_feature(to_remove)
            else:
                subclass_layer = None
                sc_svms = None
                subclass_layer = get_subclass_neurons(svm_multiplier_1, svm_cost_2, np.inf, misclass_tolerance, strictness)

        print('End subconcepts:',time.time())
        return subclass_layer

    def get_subclass_weights(sc_i, diff_used=None, svm_cost_2=None, margin_ratio=None, misclass_tolerance=None, strictness=0, level=0, subclass_layer=None):
        """Find the weights for a given subconcept neuron"""
        
        if subclasses[sc_i].level != level:
            return None

        if svm_cost_2 is None:
            svm_cost_2 = parameters['svm_cost_2']
        if margin_ratio is None:
            margin_ratio = parameters['margin_ratio']
        if misclass_tolerance is None:
            misclass_tolerance = parameters['misclass_tolerance']

        sc = subclasses[sc_i]
        num_diff = differentiae.num_nodes()
        if parameters['verbose']:
            print(sc_i, end='')
        
        #Get the relevant differentia neurons that will be used
        if not np.isinf(margin_ratio) and parameters['prune_level']<=0:
            diff_ind = differentiae.get_subclass_diff(sc_i)
        else: 
            diff_ind = range(num_diff)
        
        #Keep track of which differentiae have been kept by other subconcept neurons
        if diff_used is None:
            diff_used = np.zeros(num_diff)
        
        #Get all points to compare against the given subclass
        if parameters['intraclass_diff']:
            interclass_points = np.setdiff1d(np.arange(y_train.shape[0]), sc.points)
        elif parameters['multilabel']:
            interclass_points = []
            for j in range(y_train.shape[1]):
                if sc.y_class[0][j] == 0 or sc.y_class[0][j]==1:
                    interclass_points.extend([i for i in range(y_train.shape[0]) if y_train[i,j] != sc.y_class[0][j]])
            interclass_points = np.unique(np.array(interclass_points))
            interclass_points = [i for i in range(y_train.shape[0]) if not np.all(y_train[i, :]==sc.y_class[0])]
        else:
            interclass_points = [i for i in range(y_train.shape[0]) if y_train[i] != sc.y_class[0]]
        print("({}) vs. ({})".format(len(interclass_points),len(sc.points)), end=' ' )
        
        if parameters['ignore_interclass_duplicates']:
            to_remove = []
            for i in range(len(interclass_points)):
                for j in range(len(sc.points)):
                    if np.all(x_train[interclass_points[i]]==x_train[sc.points[j]]):
                        to_remove.append(i)
            interclass_points = np.delete(interclass_points, to_remove)
        
        #Get the differentia outputs and train the SVM
        diff_output_sc = differentiae.compute_output(x_train[sc.points,:], use_nodes=diff_ind)
        diff_output_ic = differentiae.compute_output(x_train[interclass_points, :], use_nodes=diff_ind)

        if parameters['hierarchical_subclasses'] and level>0:
            subc_output_sc = subclass_layer.compute_output(diff_output_sc)
            subc_output_ic = subclass_layer.compute_output(diff_output_ic)
            diff_output_sc = np.hstack((diff_output_sc, subc_output_sc))
            diff_output_ic = np.hstack((diff_output_ic, subc_output_ic))
        
        if False:#True:
            final_count = min(len(diff_output_ic), 10000)
            if final_count*1.5<len(diff_output_ic):
                sc_mean = np.mean(diff_output_sc, axis=0)
                num_removals = 10
                for trial in range(num_removals):
                    ic_mean = np.mean(diff_output_ic, axis=0)
                    h = (sc_mean-ic_mean).reshape(-1, 1)
                    ic_dist = (diff_output_ic @ h).flatten()

                    num_to_remove = int((len(diff_output_ic)-final_count)/(num_removals-trial))
                    ind_to_remove = np.argpartition(ic_dist, num_to_remove)[:num_to_remove]
                    diff_output_ic = np.delete(diff_output_ic, ind_to_remove, axis=0)
            final_count = min(len(diff_output_sc), 10000)
            if final_count*1.5<len(diff_output_sc):
                ic_mean = np.mean(diff_output_ic, axis=0)
                num_removals = 10
                for trial in range(num_removals):
                    sc_mean = np.mean(diff_output_sc, axis=0)
                    h = (ic_mean-sc_mean).reshape(-1, 1)
                    sc_dist = (diff_output_sc @ h).flatten()
                    num_to_remove = int((len(diff_output_sc)-final_count)/(num_removals-trial))
                    ind_to_remove = np.argpartition(sc_dist, num_to_remove)[:num_to_remove]
                    diff_output_sc = np.delete(diff_output_sc, ind_to_remove, axis=0)

        svm = SVM(diff_output_sc, diff_output_ic, svm_cost_2, fast=parameters['fast_svm'], symbolic=parameters['symbolic'], strictness=strictness)
        svm.train()

        #Get the SVM's starting point before pruning
        if sc.first_margin is None:
            sc.first_margin = svm.get_margin()
            sc.first_misclass = svm.get_misclass_error()

        #If we want to see which differentiae aren't so necessary, let's prune them
        if (not np.isinf(margin_ratio)) or parameters['full_prunes']>0:
            feature_order = resort_features(svm, sc_i, diff_ind)
            svm = prune_svm(svm, margin_ratio, diff_used[diff_ind], misclass_tolerance, sc.first_margin, sc.first_misclass, feature_order)
            #svm = prune_svm(svm, margin_ratio, np.zeros(len(diff_ind)), misclass_tolerance, sc.first_margin, sc.first_misclass, feature_order)
            new_diff_used = [diff_ind[i] for i in svm.features]
            diff_used[new_diff_used] = True
            svm.issparse = differentiae.issparse
            svm.de_featurize(num_diff, diff_ind)
            svm.trim()

            if parameters['verbose']:
                print(' ', end='')
                for d in new_diff_used:
                    scs = differentiae.get_diff_subclasses(d)[0]
                    if scs[0]==sc_i:
                        print(scs[1], end=',')
                    else:
                        print(scs[0], end=',')
        else:
            #No pruning, so just save the subclass SVMs
            if differentiae.issparse:
                svm.weights = sparse.csc_matrix(svm.weights)
            svm.trim()
            interclass_sc = []
            if parameters['multilabel']:
                for j in range(y_train.shape[1]):
                    if sc.y_class[0][j] == 0 or sc.y_class[0][j]==1:
                        interclass_sc.extend([k for k,sc_k in enumerate(subclasses) if sc_k.y_class[0][j] != sc.y_class[0][j]])
                interclass_sc = np.unique(np.array(interclass_sc))
            for i in interclass_sc:
                if parameters['verbose']:
                    print(i, end=', ')
        if parameters['verbose']:
            print()
        return (svm, diff_used)

    def resort_features(svm, sc, diff_ind):
        """Sorts the svm's features"""
        feature_sc = np.zeros(len(diff_ind)).astype(int)
        order = np.arange(len(diff_ind))
        for i in svm.features:
            scs = differentiae.get_diff_subclasses(diff_ind[i])[0]
            if scs[0]==sc:
                feature_sc[i] = scs[1]
            else:
                feature_sc[i] = scs[0]
        to_add = np.max(feature_sc)
        feature_sc[feature_sc<sc] += 1 + to_add
        order = np.argsort(feature_sc)[::-1]
        return order

    def prune_svm(svm, margin_ratio, diff_used, tol, first_margin=None, first_misclass=None, feature_order=None):
        """Sets features of svm to zero if they are not necessary for separation"""
        if first_margin is None:
            first_margin = svm.get_margin()
            first_misclass = svm.get_misclass_error()
        if feature_order is None:
            feature_order = np.arange(len(svm.features))
        diff_ind = np.array(range(len(diff_used)))
        num_to_remove = max(1, math.floor(.2*len(diff_ind)))
        if len(diff_ind)<=100:
            num_to_remove = min(10, num_to_remove)
        cur_margin = first_margin
        prune_dynamic = True #When True, this allows for more neurons to be removed at once if they have the same weight

        min_margin = first_margin*margin_ratio
        if np.isinf(margin_ratio):
            min_margin = -np.inf
        #Successivly prune away little-used differentia neurons
        while True:
            if parameters['verbose']:
                print(' -', len(diff_ind), '(', np.round(cur_margin*2, 3), ')', end='')
            if len(diff_ind) <= num_to_remove:
                break
            #Get differentiae to try to remove
            ind_to_remove = get_ind_to_remove(svm, num_to_remove, diff_used[diff_ind], feature_order, dynamic=prune_dynamic)
            
            #Remove the differentiae in a copy of the SVM and train it
            new_diff = np.delete(diff_ind, feature_order[ind_to_remove])
            new_svm = svm.copy()
            new_svm.set_features(new_diff)
            new_svm.train()
            cur_margin = new_svm.get_margin()

            #If the new SVM's margin is too low or its error as increased too much, then try to remove fewer neurons
            if new_svm.get_margin()<min_margin or new_svm.get_misclass_error()>(first_misclass*(1-tol)+tol):
                if num_to_remove>1:
                    if prune_dynamic:
                        prune_dynamic = False
                    else:
                        if differentiae.issparse:
                            num_to_remove = math.floor(num_to_remove*.9)
                        else:
                            num_to_remove = math.floor(num_to_remove/2)
                        if num_to_remove<1:
                            num_to_remove = 1
                    continue
                #We cannot remove anymore, so finish pruning
                if parameters['verbose']:
                    print(' -', len(diff_ind), '(', np.round(cur_margin*2, 3), ')', end='')
                break
            #Attempted pruning was successful, so go ahead and do it
            svm = new_svm
            for i in ind_to_remove:
                feature_order[feature_order>feature_order[i]] -= 1
            feature_order = np.delete(feature_order, ind_to_remove)
            diff_ind = new_diff
            if differentiae.issparse:
                num_to_remove = min(math.floor(len(diff_ind)*.9), num_to_remove)
            else:
                num_to_remove = min(math.floor(len(diff_ind)/2), num_to_remove)
            if num_to_remove<1:
                num_to_remove=1
            if len(diff_ind)==1:
                if parameters['verbose']:
                    print(' -', len(diff_ind), '(', np.round(cur_margin*2, 3), ')', end='')
                break

        return svm

    def get_concept_layer(svm_multiplier_2, strictness=0):
        """Gets the last ENN layer; does so by computing SVMs and then doing gradient descent to
        scale the weights so that outputs are probabilities"""
        print('Start Concepts:',time.time())
        num_epochs = 30000

        num_samples = x_train.shape[0]
        num_subclasses = subclass_layer.num_nodes()
        num_classes = len(parameters['classes'])
        subclass_pre_output = subclass_layer.compute_output(differentiae.compute_output(x_train), False)
        
        #We will initialize the concept layer with SVMs and then define what the target output should look like
        if parameters['regression']:
            concept_layer = Layer('linear')
            target = np.reshape(y_train, (-1, 1))
            concept_layer.weights = np.random.rand(num_subclasses, 1)
            concept_layer.biases = np.random.rand(1, 1)
        elif parameters['multilabel']:
            #The case where we want multiple output neurons to fire
            target = y_train.copy()
            if parameters['softmax']:
                concept_layer = Layer('softmax')
                for i in range(num_samples):
                    target[i,:] /= np.sum(target[i,:])
            else:
                concept_layer = Layer('tanh')
            concept_layer.multilabel = True
            mult_factor = svm_multiplier_2
            num_classes = y_train.shape[1]
            
            for c in range(num_classes):
                svm = SVM(subclass_layer.activation(subclass_pre_output*mult_factor), (y_train[:,c]==1).astype(int)*2-1, 10, fast=parameters['fast_svm'], strictness=strictness)
                svm.train()
                if parameters['direct_subclasses']:
                    #Don't use SVM but instead make direct connections between concept neurons and their subconcept neurons
                    target_output = target[:, c]
                    if parameters['softmax']:
                        target_output[target_output==1] = 1/(1+math.exp(-20))
                        target_output[target_output==0] = 1/(1+math.exp(20))
                        target_output = .5*np.log(target_output/(1-target_output)).reshape((-1, 1))
                        target_output = np.maximum(-10, np.minimum(10, target_output))
                        subclass_output = subclass_layer.activation(subclass_pre_output*mult_factor)
                        sc_ind = [sc.points[0] for sc in subclasses]
                        weights = np.linalg.solve(subclass_output[sc_ind, :], target_output[sc_ind, :])
                        found_weights = np.allclose(subclass_output@weights, target_output)
                    else:
                        target_output[target_output==1] = 1/(1+math.exp(-10))
                        target_output[target_output==0] = 1/(1+math.exp(10))
                        target_output = np.log(target_output/(1-target_output)).reshape((-1, 1))
                        subclass_output = subclass_layer.activation(subclass_pre_output*mult_factor)
                        sc_ind = [sc.points[0] for sc in subclasses]
                        weights = np.linalg.solve(subclass_output[sc_ind, :], target_output[sc_ind, :])
                        found_weights = np.allclose(subclass_output@weights, target_output)
                    if found_weights:
                        svm.weights = weights
                        svm.bias = [0.0]
                    else:
                        svm.weights = np.zeros(svm.weights.shape)
                        for sc_i,sc in enumerate(subclasses):
                            if sc.y_class[0][c]==1:
                                weight = 10
                            elif sc.y_class[0][c]==0:
                                weight = -10
                            else:
                                weight = math.log(sc.y_class[0][c]/(1-sc.y_class[0][c]))
                                weight = min(10, max(-10, weight))
                            svm.weights[sc_i] = weight
                        svm.bias = [np.sum(svm.weights)]
                concept_layer.add_node(svm)
        else:
            #The usual case of generating a single output label
            if parameters['softmax']:
                concept_layer = Layer('softmax')
            else:
                concept_layer = Layer('tanh', symbolic=parameters['symbolic'])
            mult_factor = svm_multiplier_2

            #The target matrix of outputs
            target = np.ones((num_samples, num_classes))*(1-parameters['concept_certainty'])/(len(parameters['classes'])-1)
            for sample in range(num_samples):
                target[sample, np.where(parameters['classes']==y_train[sample])[0]] = parameters['concept_certainty']
            #For each class train an one-vs-all SVM
            for c in parameters['classes']:
                svm = SVM(subclass_layer.activation(subclass_pre_output*mult_factor), y_train==c, 1000, fast=False, strictness=strictness)
                svm.train()
                if parameters['direct_subclasses']:
                    svm.weights = np.zeros(svm.weights.shape)
                    svm.bias = [-5.]
                    for sc_i,sc in enumerate(subclasses):
                        svm.weights[sc_i] = (sc.y_class[0]==c)*10.
                concept_layer.add_node(svm)
            concept_layer.set_mult_factor(8)
            if not parameters['direct_subclasses']:
                if parameters['symbolic']:
                    concept_layer.biases *= 16
                    concept_layer.weights *= 16
        eta_m = 0.00001
        beta_m = 0.9
        delta_m = 0

        #We will perform some gradient descent because the SVMs might not generate fully useful probabilities
        #after going through softmax
        err_previous = np.inf
        err_last_check = np.inf
        best_mult_factor = mult_factor
        best_layer = concept_layer.copy()
        stagnation_count = 0
        learn_factor = .001
        learn_factor_min = 1e-3
        print()
        print('Training concept neurons')
        print()
        if parameters['verbose']:
            print('Concept Error:', end=' ')
        
        change_back = False
        if not parameters['direct_subclasses'] and subclass_layer.activation == 'tanh':
            subclass_layer.activation = 'sigmoid'
            concept_layer.tanh_to_sigmoid()
            change_back = True
        for epoch in range(num_epochs):
            #Forward pass
            subclass_output = subclass_layer.activation(subclass_pre_output*mult_factor)
            class_output = concept_layer.compute_output(subclass_output)

            #Get error, and see if it is decreasing
            if True or parameters['regression'] or not parameters['softmax'] or parameters['direct_subclasses']:
                err = np.sqrt(np.sum((class_output-target)**2))
            else:
                err = -np.sum(np.sum(np.multiply(target, np.log(class_output))))
            if err<err_previous:
                #store best layer
                stagnation_count = 0
                if err/err_previous>.99:
                    learn_factor *= 1.05
                err_previous = err
                best_mult_factor = mult_factor
                best_layer = concept_layer.copy()
                
            else:
                stagnation_count += 1
                if learn_factor>learn_factor_min:
                    learn_factor *= .95
            
            #Print error progress
            if (epoch%200)==0:
                if parameters['verbose']:
                    print(round(err, 2), end="-")
                if err > err_last_check*0.995:
                    stagnation_count = np.inf
                err_last_check = err
            
            #If no more decreasing in error, break
            if (stagnation_count>20 and epoch>1000) or err<num_samples*1e-7:
                if parameters['verbose']:
                    print(err_previous)
                break

            #Back-propagation to do gradient descent
            if parameters['softmax']:
                dJdo = (class_output-target)
            else:
                dJdo = (class_output-target)*(class_output * (1-class_output))
            dJdw = np.matmul(subclass_output.transpose(), dJdo)
            dJdb = np.sum(dJdo, axis=0)


            learn_rate = learn_factor*np.max(np.absolute(concept_layer.weights))/np.max(np.absolute(dJdw))
            #learn_rate = learn_factor*np.mean(np.abs(concept_layer.weights[dJdw!=0]/dJdw[dJdw!=0]))
            if parameters['direct_subclasses']:
                concept_layer.weights = (concept_layer.weights - learn_rate*dJdw)*(concept_layer.weights!=0)
            else:
                concept_layer.weights = (concept_layer.weights - learn_rate*dJdw)
            concept_layer.biases = concept_layer.biases - learn_rate*dJdb

            #This allows the svm_multiplier for the subconcept layer to be changed with GD
            if parameters['learn_multiplier'] and False:
                dJds = np.matmul(dJdo, concept_layer.weights.transpose())   
                if subclass_layer.activation_function == 'sigmoid':
                    dsdm = np.multiply(np.multiply(subclass_output, 1-subclass_output), subclass_pre_output)
                else:
                    dsdm = np.multiply(np.multiply(1+subclass_output, 1-subclass_output), subclass_pre_output)*.5
                dJdm = np.sum(np.sum(np.multiply(dJds, dsdm)))
                delta_m = eta_m*dJdm*(1-beta_m) + beta_m*delta_m
                mult_factor = mult_factor - delta_m
                mult_factor = max(.01, min(svm_multiplier_2, mult_factor))
                alpha = math.log(.2)/1000
                mult_factor += alpha*(svm_multiplier_2 - mult_factor)           
        
        subclass_layer.set_mult_factor(best_mult_factor)
        
        if change_back:
            subclass_layer.activation = 'tanh'
            concept_layer.sigmoid_to_tanh()
        print('End concepts:',time.time())
        return best_layer


    def check_parameters():
        """Checks whether the parameters are valid and sets missing ones to default values"""
        if not check_key(parameters, 'verbose'):
            parameters['verbose'] = False
        if not x_train.shape[0] == y_train.shape[0]:
            print('Number of columns in training data (', x_train.shape[0], 
                  ') must match that of training labels (', y_train.shape[0], ')')
            return False
        if not check_key(parameters, 'num_subclasses'):
            return False
        if not check_key(parameters, 'hierarchical_subclasses'):
            parameters['hierarchical_subclasses'] = False
            if parameters['verbose']:
                print('Set hierarchical_subclasses to', parameters['hierarchical_subclasses'])
        if not check_key(parameters, 'superclasses'):
            parameters['superclasses'] = False
            if parameters['verbose']:
                print('Set superclasses to', parameters['superclasses'])
        if not check_key(parameters, 'suppl_coarsegrained_subconcepts'):
            parameters['suppl_coarsegrained_subconcepts'] = False
            if parameters['verbose']:
                print('Set suppl_coarsegrained_subconcepts to', parameters['suppl_coarsegrained_subconcepts'])
        if not check_key(parameters, 'suppl_unsupervised_subconcepts'):
            parameters['suppl_unsupervised_subconcepts'] = False
            if parameters['verbose']:
                print('Set suppl_unsupervised_subconcepts to', parameters['suppl_unsupervised_subconcepts'])
        if not check_key(parameters, 'unsupervised_subc_holdout'):
            parameters['unsupervised_subc_holdout'] = False
            if parameters['verbose']:
                print('Set unsupervised_subc_holdout to', parameters['unsupervised_subc_holdout'])
        if not check_key(parameters, 'alternative_subclasses'):
            parameters['alternative_subclasses'] = False
            if parameters['verbose']:
                print('Set alternative_subclasses to', parameters['alternative_subclasses'])
        if not check_key(parameters, 'parallel'):
            parameters['parallel'] = False
            if parameters['verbose']:
                print('Set parallel to', parameters['parallel'])
        if not check_key(parameters, 'clustering_weights'):
            parameters['clustering_weights'] = None
            if parameters['verbose']:
                print('Set clustering_weights to', parameters['clustering_weights'])
        if not check_key(parameters, 'cross_val_fold'):
            parameters['cross_val_fold'] = 0
            if parameters['verbose']:
                print('Train on full data set (i.e. no cross-validation)')
        if not check_key(parameters, 'fast_svm'):
            parameters['fast_svm'] = False
            if parameters['verbose']:
                print('Set fast_svm to', parameters['fast_svm'])
        if not check_key(parameters, 'softmax'):
            parameters['softmax'] = True
            if parameters['verbose']:
                print('Set softmax to', parameters['softmax'])
        if not check_key(parameters, 'prune_level'):
            parameters['prune_level'] = 0
            if parameters['verbose']:
                print('Set prune_level to', parameters['prune_level'])
        if not check_key(parameters, 'margin_ratio'):
            parameters['margin_ratio'] = 1
            if parameters['verbose']:
                print('Set margin_ratio to', parameters['margin_ratio'])
        if not check_key(parameters, 'svm_cost_1'):
            parameters['svm_cost_1'] = 15
            if parameters['verbose']:
                print('Set svm_cost_1 to', parameters['svm_cost_1'])
        if not check_key(parameters, 'svm_multiplier_1'):
            parameters['svm_multiplier_1'] = 8
            if parameters['verbose']:
                print('Set svm_multiplier_1 to', parameters['svm_multiplier_1'])
        if not check_key(parameters, 'svm_multiplier_2'):
            parameters['svm_multiplier_2'] = 1
            if parameters['verbose']:
                print('Set svm_multiplier_2 to', parameters['svm_multiplier_2'])
        if parameters['svm_multiplier_1']>16:
            parameters['symbolic'] = True
        else:
            parameters['symbolic'] = False
        if parameters['verbose']:
            print('Set symbolic to', parameters['symbolic'])
        if not check_key(parameters, 'd_strictness'):
            parameters['d_strictness'] = 0
            if parameters['verbose']:
                print('Set d_strictness to', parameters['d_strictness'])
        if not check_key(parameters, 'sc_strictness'):
            parameters['sc_strictness'] = 0
            if parameters['verbose']:
                print('Set sc_strictness to', parameters['sc_strictness'])
        if not check_key(parameters, 'c_strictness'):
            parameters['c_strictness'] = 0
            if parameters['verbose']:
                print('Set c_strictness to', parameters['c_strictness'])
        if not check_key(parameters, 'full_prunes'):
            parameters['full_prunes'] = 0
            if parameters['verbose']:
                print('Set full_prunes to', parameters['full_prunes'])
        if not check_key(parameters, 'learn_multiplier'):
            parameters['learn_multiplier'] = True
            if parameters['verbose']:
                print('Set learn_multiplier to', parameters['learn_multiplier'])
        if not check_key(parameters, 'svm_cost_2'):
            parameters['svm_cost_2'] = 10000
            if parameters['verbose']:
                print('Set svm_cost_2 to', parameters['svm_cost_2'])
        if not check_key(parameters, 'misclass_tolerance'):
            parameters['misclass_tolerance'] = 1
            if parameters['verbose']:
                print('Set misclass_tolerance to', parameters['misclass_tolerance'])
        if not check_key(parameters, 'concept_certainty'):
            parameters['concept_certainty'] = 1
            if parameters['verbose']:
                print('Set concept_certainty to', parameters['concept_certainty'])
        if not check_key(parameters, 'direct_differentiae'):
            parameters['direct_differentiae'] = False
            if parameters['verbose']:
                print('Set direct_differentiae to', parameters['direct_differentiae'])
        if not check_key(parameters, 'direct_subclasses'):
            parameters['direct_subclasses'] = False
            if parameters['verbose']:
                print('Set direct_subclasses to', parameters['direct_subclasses'])
        if not check_key(parameters, 'regression'):
            parameters['regression'] = False
            if parameters['verbose']:
                print('Set regression to', parameters['regression'])
        if not check_key(parameters, 'standardize_inputs'):
            parameters['standardize_inputs'] = False
            if parameters['verbose']:
                print('Set standardize_inputs to', parameters['standardize_inputs'])
        if not check_key(parameters, 'intraclass_diff'):
            parameters['intraclass_diff'] = False
            if parameters['verbose']:
                print('Set intraclass_diff to', parameters['intraclass_diff'])
        if not check_key(parameters, 'multilabel'):
            parameters['multilabel'] = len(y_train.shape)>1
            if parameters['verbose']:
                print('Set multilabel to', parameters['multilabel'])
        if not check_key(parameters, 'ignore_interclass_duplicates'):
            parameters['ignore_interclass_duplicates'] = False
            if parameters['verbose']:
                print('Set ignore_interclass_duplicates to', parameters['ignore_interclass_duplicates'])
        if not check_key(parameters, 'reuse_dual_coeff'):
            parameters['reuse_dual_coeff'] = False
            if parameters['verbose']:
                print('Set reuse_dual_coeff to', parameters['reuse_dual_coeff'])
        if not check_key(parameters, 'convolutional'):
            parameters['convolutional'] = False
        elif parameters['convolutional']:
            if not check_key(parameters, 'convolution_num'):
                parameters['convolution_num'] = 6
                if parameters['verbose']:
                    print('Set convolution_num to', parameters['convolution_num'])
            if not check_key(parameters, 'convolution_size'):
                parameters['convolution_size'] = 5
                if parameters['verbose']:
                    print('Set convolution_size to', parameters['convolution_size'])
            if not check_key(parameters, 'convolution_mult'):
                parameters['convolution_mult'] = 1
                if parameters['verbose']:
                    print('Set convolution_mult to', parameters['convolution_mult'])
            if not check_key(parameters, 'convolution_padding'):
                parameters['convolution_padding'] = True
                if parameters['verbose']:
                    print('Set convolution_padding to', parameters['convolution_padding'])
            if not check_key(parameters, 'convolution_pooling'):
                parameters['convolution_pooling'] = 2
                if parameters['verbose']:
                    print('Set convolution_pooling to', parameters['convolution_pooling'])
            if not check_key(parameters, 'convolution_stride'):
                parameters['convolution_stride'] = 1
                if parameters['verbose']:
                    print('Set convolution_stride to', parameters['convolution_stride'])
            if not check_key(parameters, 'convolution_strict'):
                parameters['convolution_strict'] = np.zeros(len(parameters['convolution_size'])).tolist()
                if parameters['verbose']:
                    print('Set convolution_strict to', parameters['convolution_strict'])
            if isinstance(parameters['convolution_num'], int):
                parameters['convolution_num'] = [parameters['convolution_num']]
            if isinstance(parameters['convolution_size'], int):
                parameters['convolution_size'] = [parameters['convolution_size']]
            if isinstance(parameters['convolution_mult'], int):
                parameters['convolution_mult'] = [parameters['convolution_mult']]
            if isinstance(parameters['convolution_padding'], int):
                parameters['convolution_padding'] = [parameters['convolution_padding']]
            if isinstance(parameters['convolution_pooling'], int):
                parameters['convolution_pooling'] = [parameters['convolution_pooling']]
            if isinstance(parameters['convolution_stride'], int):
                parameters['convolution_stride'] = [parameters['convolution_stride']]
            if isinstance(parameters['convolution_strict'], int):
                parameters['convolution_strict'] = [parameters['convolution_strict']]
        parameters['grid_search'] = False
        for key, value in parameters.items():
            if isinstance(value, list):
                if not (key=='convolution_size' or key=='convolution_num' or key=='convolution_mult' or key=='convolution_stride'
                        or key=='convolution_pooling' or key=='convolution_padding' or key=='convolution_strict'):
                    parameters['grid_search'] = True
                    break
        if parameters['grid_search']:
            if not isinstance(parameters['svm_cost_1'], list):
                parameters['svm_cost_1'] = [parameters['svm_cost_1']]
            if not isinstance(parameters['svm_cost_2'], list):
                parameters['svm_cost_2'] = [parameters['svm_cost_2']]
            if not isinstance(parameters['svm_multiplier_1'], list):
                parameters['svm_multiplier_1'] = [parameters['svm_multiplier_1']]
            if not isinstance(parameters['svm_multiplier_2'], list):
                parameters['svm_multiplier_2'] = [parameters['svm_multiplier_2']]
            if not isinstance(parameters['margin_ratio'], list):
                parameters['margin_ratio'] = [parameters['margin_ratio']]
            else:
                parameters['margin_ratio'] = np.sort(parameters['margin_ratio'])[::-1].tolist()
            if not isinstance(parameters['misclass_tolerance'], list):
                parameters['misclass_tolerance'] = [parameters['misclass_tolerance']]
        parameters['training_samples'] = x_train.shape[0]
        parameters['data_dimension'] = x_train.shape[1]
        if parameters['regression']:
            parameters['classes'] = np.reshape(np.arange(int(parameters['regression'])), (1, -1))
        elif not parameters['multilabel']:
            parameters['classes'] = np.unique(y_train)
        else:
            parameters['classes'] = np.unique(y_train, axis=0)
            
        return True

    def check_key(dictionary, key):
        """Checks the precence of a key in a dictionary"""
        if not key in dictionary.keys():
            if parameters['verbose']:
                print('Missing parameter', key)
            return False
        return True

    def get_subclasses_alternative_old(num_subclasses):
        """Uses an alternative approach to define subclasses"""

        subclasses = []
        for c in parameters['classes']:
            new_subclass = Subclass(y_class=[c, 0])
            if parameters['regression']:
                new_subclass.reset_points([i for i, y in enumerate(bin_id) if y==c])
            elif parameters['multilabel']:
                new_subclass.reset_points([i for i, y in enumerate(y_train) if np.all(y==c)])
            else:
                new_subclass.reset_points([i for i, y in enumerate(y_train) if y==c])
            subclasses.append(new_subclass)
        
        #subclasses = get_overlapping_subclasses(subclasses)

        class_dist = np.full((len(subclasses), len(subclasses)), np.inf)
        class_errors = -np.ones((len(subclasses), len(subclasses)))
        
        while len(subclasses) < 3*num_subclasses:
            no_error = True
            for i in range(len(subclasses)):
                for j in range(i+1, len(subclasses)):
                    if class_errors[i,j] >= 0:
                        continue
                    best_error, best_margin = get_error_and_distance(subclasses[i].points, subclasses[j].points)
                    class_dist[i,j] = best_margin
                    class_errors[i,j] = best_error
                    if best_error>0.02:
                        no_error = False
            
            if len(subclasses)>=num_subclasses and max(class_errors.flatten())<=0.02:
                break
            
            all_dist = class_dist.flatten()
            #max_dist = np.max(all_dist[all_dist < np.inf])
            score = class_errors# - class_dist/(max_dist*2)
            inds = np.unravel_index(np.argmax(score), score.shape)
            
            """
            for i in inds:    
                if parameters['clustering_weights'] is None:
                    tree = scipy.cluster.hierarchy.linkage(x_train[subclasses[i].points, :], method='ward')
                else:
                    data = x_train[subclasses[i].points, :]
                    data = data * np.tile(parameters['clustering_weights'].reshape(1,-1), (len(subclasses[i].points),1))
                    tree = scipy.cluster.hierarchy.linkage(data , method='ward')
                clusters = scipy.cluster.hierarchy.cut_tree(tree, n_clusters=2)
                for s in range(2):
                    cluster_indices = np.where(clusters==s)[0]
                    subclasses.append(Subclass(subclasses[i].y_class, [subclasses[i].points[j] for j in cluster_indices]))
            """
            if parameters['clustering_weights'] is None:
                svm = SVM(np.mean(x_train[subclasses[inds[0]].points,:], axis=0).reshape(1,-1), np.mean(x_train[subclasses[inds[1]].points,:], axis=0).reshape(1,-1), 10, symbolic=parameters['symbolic'], fast=True)
            else:
                svm = SVM((np.mean(x_train[subclasses[inds[0]].points,:], axis=0)*parameters['clustering_weights']).reshape(1,-1),
                          (np.mean(x_train[subclasses[inds[1]].points,:], axis=0)*parameters['clustering_weights']).reshape(1,-1),
                          10, symbolic=parameters['symbolic'], fast=True)
            if np.any(svm.x_train[0,:]!=svm.x_train[1,:]):
                svm.train()
            if np.all(svm.weights==0):#max(len(subclasses[inds[0]].points),len(subclasses[inds[0]].points))>20000:
                if parameters['clustering_weights'] is None:
                    svm = SVM(x_train[subclasses[inds[0]].points,:], x_train[subclasses[inds[1]].points,:], .01, fast=True)
                else:
                    w0 = np.tile(parameters['clustering_weights'].reshape(1,-1), (len(subclasses[inds[0]].points),1))
                    w1 = np.tile(parameters['clustering_weights'].reshape(1,-1), (len(subclasses[inds[1]].points),1))
                    svm = SVM(x_train[subclasses[inds[0]].points,:]*w0, x_train[subclasses[inds[1]].points,:]*w1, .000001, fast=True)
                svm.train()
            if np.all(np.abs(svm.weights)<1e-6):
                if parameters['clustering_weights'] is None:
                    svm.weights = np.ones((x_train.shape[1],1))
                    svm.weights = np.random.randint(0,2,(x_train.shape[1],1))
                else:
                    svm.weights = parameters['clustering_weights'].reshape(-1,1)
            dist0 = (x_train[subclasses[inds[0]].points,:] @ svm.weights).flatten()
            dist1 = (x_train[subclasses[inds[1]].points,:] @ svm.weights).flatten()
            all_dist = np.round(np.concatenate((dist0, dist1)),6)
            all_labels = np.ones(len(all_dist))
            all_labels[:len(dist0)] = 0
            order = np.argsort(all_dist)
            all_labels = all_labels[order]
            all_dist = all_dist[order]

            num0_above = len(dist0)
            num1_above = len(dist1)
            num0_below = 0
            num1_below = 0
            
            best_bias = 0
            best_err = np.inf
            unique_dist = np.unique(all_dist)
            for b in range(len(unique_dist)):
                ind = np.where(all_dist==unique_dist[b])[0]
                num0 = np.sum(all_labels[ind]==0)
                num1 = len(ind)-num0
                num0_above -= num0
                num1_above -= num1
                err = (num0_below/len(dist0) + num1_above/len(dist1))/2
                if err <= best_err or (1-err) <= best_err:
                    best_bias = -unique_dist[b]
                    best_err = min(err,1-err)
                num0_below += num0
                num1_below += num1
            svm.bias = best_bias

            
            found_split = None
            error = True
            for i in range(2):
                dist = (x_train[subclasses[inds[i]].points,:] @ svm.weights + svm.bias)*(1-2*i)
                ind_p = [i for i,d in enumerate(dist) if d>=0]
                ind_n = [i for i,d in enumerate(dist) if d<0]
                if len(ind_p)==0 or len(ind_n)==0:
                    found_split = 1-i
                else:
                    error = False
                    subclasses.append(Subclass(subclasses[inds[i]].y_class, [subclasses[inds[i]].points[j] for j in ind_p]))
                    subclasses.append(Subclass(subclasses[inds[i]].y_class, [subclasses[inds[i]].points[j] for j in ind_n]))
            if error:
                best_bias = 0
                best_err = np.inf
                unique_dist = np.unique(all_dist)
                for b in range(len(unique_dist)-1):
                    ind = np.where(all_dist==unique_dist[b])[0]
                    num0 = np.sum(all_labels[ind]==0)
                    num1 = len(ind)-num0
                    num0_above -= num0
                    num1_above -= num1
                    num0_below += num0
                    num1_below += num1
                    err = (num0_below/len(dist0) + num1_above/len(dist1))/2
                    if err <= best_err or (1-err) <= best_err:
                        best_bias = -(unique_dist[b]+unique_dist[b+1])/2
                        best_err = min(err,1-err)
                svm.bias = best_bias

                found_split = None
                error = True
                for i in range(2):
                    dist = x_train[subclasses[inds[i]].points,:] @ svm.weights + svm.bias
                    ind_p = [i for i,d in enumerate(dist) if d>0]
                    ind_n = [i for i,d in enumerate(dist) if d<=0]
                    if len(ind_p)==0 or len(ind_n)==0:
                        found_split = 1-i
                    else:
                        error = False
                        subclasses.append(Subclass(subclasses[inds[i]].y_class, [subclasses[inds[i]].points[j] for j in ind_p]))
                        subclasses.append(Subclass(subclasses[inds[i]].y_class, [subclasses[inds[i]].points[j] for j in ind_n]))
            if error:
                found_split = None
                while error:
                    error = True
                    new_subclasses = []
                    for i in range(2):
                        dist = x_train[subclasses[inds[i]].points,:] @ svm.weights + svm.bias
                        ind_p = [i for i,d in enumerate(dist) if np.random.rand()<.5]
                        ind_n = [i for i,d in enumerate(dist) if i not in ind_p]
                        if len(ind_p)==0 or len(ind_n)==0:
                            found_split = 1-i
                            continue
                        else:
                            error = False
                            new_subclasses.append(Subclass(subclasses[inds[i]].y_class, [subclasses[inds[i]].points[j] for j in ind_p]))
                            new_subclasses.append(Subclass(subclasses[inds[i]].y_class, [subclasses[inds[i]].points[j] for j in ind_n]))
                for s in new_subclasses:
                    subclasses.append(s)
            if found_split is None:
                best_error_00, best_margin_00 = get_error_and_distance(subclasses[-4].points, subclasses[inds[1]].points, total=True)
                best_error_01, best_margin_01 = get_error_and_distance(subclasses[-3].points, subclasses[inds[1]].points, total=True)
                best_error_10, best_margin_10 = get_error_and_distance(subclasses[-2].points, subclasses[inds[0]].points, total=True)
                best_error_11, best_margin_11 = get_error_and_distance(subclasses[-1].points, subclasses[inds[0]].points, total=True)
                max_dist = np.max((best_margin_00, best_margin_01, best_margin_10, best_margin_11))
                score_00 = best_error_00# - best_margin_00/(max_dist*2)
                score_01 = best_error_01# - best_margin_01/(max_dist*2)
                score_10 = best_error_10# - best_margin_10/(max_dist*2)
                score_11 = best_error_11# - best_margin_11/(max_dist*2)
                if score_00 + score_01 > score_10 + score_11: #Splitting the first subclass is worse
                    del subclasses[-4]
                    del subclasses[-3]
                    ind_to_split = inds[1]
                else:
                    del subclasses[-2]
                    del subclasses[-1]
                    ind_to_split = inds[0]
            else:
                ind_to_split = inds[found_split]
            del subclasses[ind_to_split]
            class_dist = np.delete(class_dist, ind_to_split, axis=0)
            class_dist = np.delete(class_dist, ind_to_split, axis=1)
            class_errors = np.delete(class_errors, ind_to_split, axis=0)
            class_errors = np.delete(class_errors, ind_to_split, axis=1)
            
            class_dist = np.vstack((class_dist, np.full((2, class_dist.shape[1]), np.inf)))
            class_dist = np.hstack((class_dist, np.full((class_dist.shape[0], 2), np.inf)))
            class_errors = np.vstack((class_errors, -np.ones((2, class_errors.shape[1]))))
            class_errors = np.hstack((class_errors, -np.ones((class_errors.shape[0], 2))))
        
        print('Found {} subclasses'.format(len(subclasses)))

        return subclasses
    
    def get_overlapping_subclasses(subclasses):
        new_subclasses = []
        for s in subclasses:
            while True:
                points = [s.points[0]]
                get_more = True
                if np.all(x_train[s.points[0],:]==0):
                    if len(s.points)>1:
                        points.append(s.points[1])
                    else:
                        get_more = False
                l = len(points)
                while get_more:
                    existing = np.max(np.abs(x_train[points,:]), axis=0)
                    inds = np.where(existing!=0)[0]
                    max_val = np.max(np.abs(x_train[:,inds][s.points]), axis=1)
                    new_points = [s.points[i] for i in np.where(max_val!=0)[0]]
                    if len(new_points) <= len(points):
                        break
                    points = new_points
                new_subclasses.append(Subclass(s.y_class, points))
                for p in points:
                    s.points.remove(p)
                if len(s.points)==0:
                    break
        if len(new_subclasses)<2:
            debug=1
        return new_subclasses

    def get_error_and_distance(points_i, points_j, total=False):
        """Used for the alternate method of getting subclasses;
        gets the distance between two sets of points as well as the error in a simple separation"""
        if len(points_i)==0 or len(points_j)==0:
            return 0, 1e6
        #svm = SVM(np.mean(x_train[points_i,:], axis=0).reshape(1,-1), np.mean(x_train[points_j,:], axis=0).reshape(1,-1), 10, fast=True, symbolic=parameters['symbolic'])
        #if np.any(svm.x_train[0,:]!=svm.x_train[1,:]):
        #    svm.train()
        if True:#np.all(svm.weights==0):
            svm = SVM(x_train[points_i,:], x_train[points_j,:], 10, fast=True)
            svm.train()
        if np.all(np.abs(svm.weights)<1e-6):
            svm.weights = np.ones((x_train.shape[1],1))
        dist0 = (x_train[points_i,:] @ svm.weights).flatten()
        dist1 = (x_train[points_j,:] @ svm.weights).flatten()
        all_dist = np.concatenate((dist0, dist1))
        all_labels = np.ones(len(all_dist))
        all_labels[:len(dist0)] = 0
        order = np.argsort(all_dist)
        all_labels = all_labels[order]
        all_dist = all_dist[order]

        num0_above = len(dist0)
        num1_above = len(dist1)
        num0_below = 0
        num1_below = 0
        
        best_b = 0
        best_err = .5
        tot_err = 0
        for b in range(len(all_dist)-1):
            if all_labels[b]==0:
                num0_below += 1
                num0_above -= 1
            else:
                num1_below += 1
                num1_above -= 1
            err = (num0_below/len(dist0) + num1_above/len(dist1))/2
            if all_dist[b+1]!=all_dist[b]:
                if err < best_err or (1-err) < best_err:
                    best_b = b
                    best_err = min(err,1-err)
                    tot_err = min(num0_below+num1_above, num1_below+num0_above)
        svm.bias = -(all_dist[best_b-1]+all_dist[best_b])/2
        if total:
            return tot_err, svm.get_margin()
        return best_err, svm.get_margin()

        avg_i = np.mean(x_train[points_i,:], axis=0)
        avg_j = np.mean(x_train[points_j,:], axis=0)
        h = (avg_j-avg_i).reshape(-1,1)
        dist_i = (x_train[points_i,:] @ h).flatten()
        dist_j = (x_train[points_j,:] @ h).flatten()
        len_i = len(dist_i)
        order = np.argsort(np.concatenate((dist_i, dist_j)))
        best_error = np.inf
        best_margin = 0
        each_side = [[len(dist_i), len(dist_j)], [0, 0]]
        for k in range(len(order)-1):
            if order[k]>=len_i:
                each_side[0][1] -= 1
                each_side[1][1] += 1
            else:
                each_side[0][0] -= 1
                each_side[1][0] += 1
            new_err_1 = each_side[0][0]/len(dist_i) + each_side[1][1]/len(dist_j)
            new_err_2 = each_side[0][1]/len(dist_i) + each_side[1][0]/len(dist_j)
            new_err = min(new_err_1, new_err_2)
            if new_err<best_error:
                best_error = new_err
                if order[k]>=len_i:
                    dist_1 = dist_j[order[k]-len_i]
                else:
                    dist_1 = dist_i[order[k]]
                if order[k+1]>=len_i:
                    dist_2 = dist_j[order[k+1]-len_i]
                else:
                    dist_2 = dist_i[order[k+1]]
                best_margin = np.abs(dist_2-dist_1)
        
        return best_error, best_margin

    return train()

def get_all_differentiae_test(s_i, x_train=None, y_train=None, subclasses=None, svm_cost=15, intraclass=False, symbolic=False, strictness=0, margin_ratio=None, parameters=None):
    """This is a copy of the get_all_differentiae function at the module level in order to be able to parallelize in specific cases"""
    print()
    print(s_i, end=' - ')

    if margin_ratio is not None and (margin_ratio<100):
        prune = True
    else:
        prune = False

    if isinstance(x_train, tuple):
        file_name = x_train[0]
        #x_shape = x_train[1]
        x_train = np.load(file_name)
        print('LOADING DIFFERENTIA X_TRAIN')
        #x_train = x_train.reshape(x_shape)
    all_svms = []
    sc_indices = []
    for s_j in range(len(subclasses)):
        if s_j<=s_i and strictness==0 and (not prune):
            continue
        #print(s_j, end=' ')
        if (not intraclass) and np.all(subclasses[s_i].y_class[0]==subclasses[s_j].y_class[0]):
            continue
        num_overlapping = len(np.intersect1d(subclasses[s_i].points, subclasses[s_j].points))
        frac_overlapping = num_overlapping/min(len(subclasses[s_i].points), len(subclasses[s_j].points))
        if frac_overlapping>0:
            continue
        if s_i<s_j:
            svm = SVM(x_train[subclasses[s_i].points,:], x_train[subclasses[s_j].points,:], svm_cost, fast=True, symbolic=symbolic, strictness=strictness)
            sc_indices.append([s_i, s_j])
        else:
            svm = SVM(x_train[subclasses[s_j].points,:], x_train[subclasses[s_i].points,:], svm_cost, fast=True, symbolic=symbolic, strictness=strictness)
            sc_indices.append([s_j, s_i])
        svm.train()
        svm.trim()
        all_svms.append(svm)
        
    
    if prune:
        diff_layer = Layer(symbolic=symbolic)
        diff_layer.subclass_indices= np.zeros((len(subclasses), len(subclasses)))
        diff_layer.add_nodes(all_svms, sc_indices)
        diff_layer.set_mult_factor(parameters['svm_multiplier_1'])
        results = get_subclass_weights_test(s_i, x_train=x_train, y_train=y_train, subclasses=subclasses, differentiae=diff_layer, parameters=parameters)
        if results is not None:
            print('Subconcept {} pruned {} differentiae to {}'.format(s_i,len(results[1]),sum(results[1])))
            diff_used = results[1]
            all_svms = [all_svms[s] for s,d in enumerate(diff_used) if d]
            sc_indices = [sc_indices[s] for s,d in enumerate(diff_used) if d]

    return all_svms, sc_indices

def get_subclass_weights_test(sc_i, x_train=None, y_train=None, subclasses=None, differentiae=None, svm_cost_2=None, margin_ratio=None, misclass_tolerance=None, symbolic=True, multilabel=False, intraclass_diff=False, diff_used=None, parameters=None, verbose=True, strictness=0, level=0, hierarchical=False, subclass_layer=None, fast_svm=True, d_strictness=0):

    if parameters is not None:
        svm_cost_2 = parameters['svm_cost_2']
        margin_ratio = parameters['margin_ratio']
        misclass_tolerance = parameters['misclass_tolerance']
        symbolic = parameters['symbolic']
        multilabel = parameters['multilabel']
        intraclass_diff = parameters['intraclass_diff']
        verbose=parameters['verbose']
        strictness=parameters['sc_strictness']
        hierarchical=parameters['hierarchical_subclasses']
        fast_svm=parameters['fast_svm']
        d_strictness=parameters['d_strictness']

    if hierarchical and subclasses[sc_i].level != level:
        return None

    already_computed = False
    used_direct = False
    if isinstance(x_train, (list, tuple)):
        if len(x_train)==2:
            r = x_train[0]
            file_name = 'x_train_' + str(r) + '.npy'
            x_shape = x_train[1]
            x_train = np.load(file_name)
            x_train = x_train.reshape(x_shape)
            print('LOADING SUBCLASS X_TRAIN')
            
            file_name = 'diff_' + str(r) + '.npz'
            differentiae = Layer.load(file_name)
        else:
            already_computed = True
            r = x_train[0]
            print('Loading SUBCLASS DIFF_OUTPUT', sc_i)
            if hierarchical and level>0:
                file_name = 'subc_output_' + str(r) + '.npy'
                subc_output = np.load(file_name)
            if x_train[2] is not None:
                file_name = 'diff_output_' + str(r) + '_' + str(sc_i) + '.npy'
                used_direct = True
            else:
                file_name = 'diff_output_' + str(r) + '.npy'
                x_shape = x_train[1]
            diff_output = np.load(file_name)

    if svm_cost_2 is None:
        svm_cost_2 = parameters['svm_cost_2']
    if margin_ratio is None:
        margin_ratio = parameters['margin_ratio']
    if misclass_tolerance is None:
        misclass_tolerance = parameters['misclass_tolerance']
    
    sc = subclasses[sc_i]
    if not already_computed:
        num_diff = differentiae.num_nodes()
    else:
        num_diff = diff_output.shape[1]
    if verbose:
        print(sc_i, end='')
    if margin_ratio<100 and not already_computed:
        diff_ind = differentiae.get_subclass_diff(sc_i, d_strictness==0)
    else:
        diff_ind = range(num_diff)
    if diff_used is None:
        diff_used = np.zeros(num_diff)
    if intraclass_diff:
        interclass_points = np.setdiff1d(np.arange(y_train.shape[0]), sc.points)
    elif multilabel:
        interclass_points = []
        for j in range(y_train.shape[1]):
            if sc.y_class[0][j] == 0 or sc.y_class[0][j]==1:
                interclass_points.extend([i for i in range(y_train.shape[0]) if y_train[i,j] != sc.y_class[0][j]])
        interclass_points = np.unique(np.array(interclass_points))
        interclass_points = [i for i in range(y_train.shape[0]) if not np.all(y_train[i, :]==sc.y_class[0])]
    else:
        interclass_points = [i for i in range(y_train.shape[0]) if y_train[i] != sc.y_class[0]]
    print("({}) vs. ({})".format(len(interclass_points),len(sc.points)), end=' ' )
    if not already_computed:
        diff_output_sc = differentiae.compute_output(x_train[sc.points,:], use_nodes=diff_ind)
        diff_output_ic = differentiae.compute_output(x_train[interclass_points, :], use_nodes=diff_ind)
    else:
        diff_output_sc = diff_output[sc.points, :][:,diff_ind]
        diff_output_ic = diff_output[interclass_points, :][:,diff_ind]

    if hierarchical and level>0:
        subc_output_sc = subc_output[sc.points, :]
        subc_output_ic = subc_output[interclass_points, :]
        diff_output_sc = np.hstack((diff_output_sc, subc_output_sc))
        diff_output_ic = np.hstack((diff_output_ic, subc_output_ic))
    
    if False:#True:
        final_count = min(len(diff_output_ic), 5000)
        if final_count*1.5<len(diff_output_ic):
            sc_mean = np.mean(diff_output_sc, axis=0)
            num_removals = 10
            for trial in range(num_removals):
                ic_mean = np.mean(diff_output_ic, axis=0)
                h = (sc_mean-ic_mean).reshape(-1, 1)
                ic_dist = (diff_output_ic @ h).flatten()
                num_to_remove = int((len(diff_output_ic)-final_count)/(num_removals-trial))
                ind_to_remove = np.argpartition(ic_dist, num_to_remove)[:num_to_remove]
                diff_output_ic = np.delete(diff_output_ic, ind_to_remove, axis=0)
        final_count = min(len(diff_output_sc), 5000)
        if final_count*1.5<len(diff_output_sc):
            ic_mean = np.mean(diff_output_ic, axis=0)
            num_removals = 10
            for trial in range(num_removals):
                sc_mean = np.mean(diff_output_sc, axis=0)
                h = (ic_mean-sc_mean).reshape(-1, 1)
                sc_dist = (diff_output_sc @ h).flatten()
                num_to_remove = int((len(diff_output_sc)-final_count)/(num_removals-trial))
                ind_to_remove = np.argpartition(sc_dist, num_to_remove)[:num_to_remove]
                diff_output_sc = np.delete(diff_output_sc, ind_to_remove, axis=0)

    """
    if symbolic:
        diff_output_ic *= 2
        diff_output_ic -= 1
        diff_output_sc *= 2
        diff_output_sc -= 1
    print()
    for diff in [diff_output_sc, diff_output_ic]:
        for i in range(len(diff)):
            for j in range(len(diff[i])):
                print(int(diff[i,j]), end=' ')
            print()
        print()
        print()
        print()
    if sc_i>0:
        a = np.zeros(4)
        a[5] = a[9]
    """
    svm = SVM(diff_output_sc, diff_output_ic, svm_cost_2, fast=int(fast_svm)+int(margin_ratio<100), symbolic=symbolic, strictness=strictness)
    svm.id = sc_i
    svm.exemplar = False
    svm.train()
    """
    for j in range(svm.weights.size):
        print(svm.weights.flatten()[j], end=' ')
    print()
    print(svm.bias)
    print()
    print()
    """
    if margin_ratio<100:
        if sc.first_margin is None:
            sc.first_margin = svm.get_margin()
            sc.first_misclass = svm.get_misclass_error()

        
        feature_sc = np.zeros(len(diff_ind)).astype(int)
        for i in svm.features:
            scs = differentiae.get_diff_subclasses(diff_ind[i])[0]
            if scs[0]==sc_i:
                feature_sc[i] = scs[1]
            else:
                feature_sc[i] = scs[0]
        to_add = np.max(feature_sc)
        feature_sc[feature_sc<sc_i] += 1 + to_add
        feature_order = np.argsort(feature_sc)[::-1]
        #feature_order = np.arange(len(svm.features))[::-1]
        #feature_order = np.argsort(np.sum(differentiae.weights[:,diff_ind]==0, axis=0)*1000 - np.argmax(differentiae.weights[:,diff_ind]!=0, axis=0))
        if isinstance(feature_order, np.matrix):
            feature_order = np.array(feature_order).flatten()
        
        #feature_order = resort_features(svm, sc_i, diff_ind)

        #svm = prune_svm_test(svm, margin_ratio, diff_used[diff_ind], misclass_tolerance, sc.first_margin, sc.first_misclass, feature_order, sparse=differentiae.issparse)
        svm = prune_svm_test(svm, margin_ratio, np.zeros(len(diff_ind)), misclass_tolerance, sc.first_margin, sc.first_misclass, feature_order, sparse=differentiae.issparse)
        new_diff_used = [diff_ind[i] for i in svm.features]
        diff_used[new_diff_used] = True
        svm.issparse = differentiae.issparse
        if already_computed:
            num_diff = x_train[2]
            diff_ind = x_train[3][sc_i]
            if hierarchical and level>0:
                new_ind = num_diff + np.arange(subc_output.shape[1])
                diff_ind = np.concatenate((diff_ind, new_ind))
                num_diff += subc_output.shape[1]
        svm.de_featurize(num_diff, diff_ind)
        svm.trim()

        if verbose:
            print(' ', end='')
            for d in new_diff_used:
                scs = differentiae.get_diff_subclasses(d)[0]
                if scs[0]==sc_i:
                    print(scs[1], end=',')
                else:
                    print(scs[0], end=',')
    else:
        if not already_computed and differentiae.issparse:
            svm.weights = sparse.csc_matrix(svm.weights)
        if used_direct:
            num_diff = x_train[2]
            diff_ind = x_train[3][sc_i]
            if hierarchical and level>0:
                new_ind = num_diff + np.arange(subc_output.shape[1])
                diff_ind = np.concatenate((diff_ind, new_ind))
                num_diff += subc_output.shape[1]
            svm.de_featurize(num_diff, diff_ind)
        svm.trim()
        interclass_sc = []
        if multilabel:
            for j in range(y_train.shape[1]):
                if sc.y_class[0][j] == 0 or sc.y_class[0][j]==1:
                    interclass_sc.extend([k for k,sc_k in enumerate(subclasses) if sc_k.y_class[0][j] != sc.y_class[0][j]])
            interclass_sc = np.unique(np.array(interclass_sc))
            #interclass_sc = [j for j,sc_j in enumerate(subclasses) if not np.all(sc_j.y_class[0]==sc.y_class[0])]
        for i in interclass_sc:
            if verbose:
                print(i, end=', ')
    if verbose:
        print()
    
    """
    if symbolic:
        svm.trim()
        svm.bias -= np.sum(svm.weights)
        svm.weights *= 2
    """
    
    return (svm, diff_used)

def prune_svm_test(svm, margin_ratio, diff_used, tol, first_margin=None, first_misclass=None, feature_order=None, sparse=False):
    """Sets features of svm to zero if they are not necessary for separation"""
    if first_margin is None:
        first_margin = svm.get_margin()
        first_misclass = svm.get_misclass_error()
    if feature_order is None:
        feature_order = np.arange(len(svm.features))
    diff_ind = np.array(range(len(diff_used)))
    num_to_remove = max(1, math.floor(.2*len(diff_ind)))
    if len(diff_ind)<=100:
        num_to_remove = min(10, num_to_remove)
    cur_margin = first_margin
    prune_dynamic = True
    while True:
        print(' -', len(diff_ind), '(', np.round(cur_margin*2, 3), ')', end='')
        if len(diff_ind) <= num_to_remove:
            break
        ind_to_remove = get_ind_to_remove(svm, num_to_remove, diff_used[diff_ind], feature_order, dynamic=prune_dynamic)
        new_diff = np.delete(diff_ind, feature_order[ind_to_remove])
        new_svm = svm.copy()
        new_svm.set_features(new_diff)
        new_svm.train()
        cur_margin = new_svm.get_margin()
        if new_svm.get_margin()<first_margin*margin_ratio or new_svm.get_misclass_error()>(first_misclass*(1-tol)+tol):
            if num_to_remove>1:
                if prune_dynamic:
                    prune_dynamic = False
                else:
                    if sparse:
                        num_to_remove = math.floor(num_to_remove*.9)
                    else:
                        num_to_remove = math.floor(num_to_remove/2)
                    if num_to_remove<1:
                        num_to_remove = 1
                continue
            print(' -', len(diff_ind), '(', np.round(cur_margin*2, 3), ')', end='')
            break
        svm = new_svm
        for i in ind_to_remove:
            feature_order[feature_order>feature_order[i]] -= 1
        feature_order = np.delete(feature_order, ind_to_remove)
        diff_ind = new_diff
        if sparse:
            num_to_remove = min(math.floor(len(diff_ind)*.9), num_to_remove)
        else:
            num_to_remove = min(math.floor(len(diff_ind)/2), num_to_remove)
        if num_to_remove<1:
            num_to_remove=1
        if len(diff_ind)==1:
            print(' -', len(diff_ind), '(', np.round(cur_margin*2, 3), ')', end='')
            break

    return svm

def get_ind_to_remove(svm, num_to_remove, diff_used, f_order, dynamic=True):
    """Gets the indices of the svm to try to remove in pruning"""
    diff_used = np.zeros(diff_used.shape)
    rounded_weights = np.absolute(svm.weights[np.array(svm.features)[f_order]].flatten())
    for i in range(len(f_order)):
        if rounded_weights[i] != 0:
            rounded_weights[i] = np.round(rounded_weights[i], -int(math.floor(math.log10(rounded_weights[i]))))

    sorted_weights_ind = np.argsort(rounded_weights, kind='mergesort')
    min_k_ind = sorted_weights_ind[:num_to_remove]
    min_k = rounded_weights[min_k_ind[-1]]
    close_diff_ind = [i for i in sorted_weights_ind if rounded_weights[i]<min_k*1.05]
    
    if dynamic and num_to_remove>1 and np.sum(rounded_weights==min_k)>1:
        if np.sum(rounded_weights<min_k)>0:
            min_k_ind = np.where(rounded_weights<min_k)[0]
            num_to_remove = len(min_k_ind)
            diff_used = np.zeros(diff_used.shape)
        elif np.sum(rounded_weights<=min_k)<len(svm.features):
            min_k_ind = np.where(rounded_weights<=min_k)[0]
            num_to_remove = len(min_k_ind)
            diff_used = np.zeros(diff_used.shape)
    
    ind_to_remove = np.full(num_to_remove, np.nan)
    curr_ind = 0
    for i in close_diff_ind:
        if diff_used[i]:
            ind_to_remove[curr_ind] = i
            curr_ind += 1
            if curr_ind == num_to_remove:
                break
    if curr_ind<num_to_remove:
        for i in min_k_ind:
            if not (i in ind_to_remove):
                ind_to_remove[curr_ind] = i
                curr_ind += 1
                if curr_ind == num_to_remove:
                    break
    return ind_to_remove.astype(int)

def train_enn(x_train, y_train, parameters, x_val=None, y_val=None, x_train_0=None):
    return train_full_enn(x_train, y_train, parameters, x_val, y_val, x_train_0)