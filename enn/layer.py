"""A Layer object that contains a layer of an ENN"""

#Imports
import numpy as np
import scipy.sparse as sparse

class Layer():
    """An ENN layer"""
    def __init__(self, act_function='tanh', convolutional=False, convolution_win_size=None, pooling=2, stride=1, padding=False, symbolic=False):
        self.weights = []
        self.biases = []
        self.activation_function = act_function
        self.subclass_indices = np.zeros((1,1)) #Stores the subclasses each differentia separates
        self.midpoints = None #Stores the midpoints of the SVMs generated for each neuron
        self.sum_dual_coef = [] #Stores information about the SVM
        self.support_vectors = [] #The SVM's support vectors
        self.mult_factor = 1 #SVMs are weighted such that SVs are weighted distance of 1; this can scale that
        self.multilabel = False
        self.symbolic = symbolic
        self.label_lengths = [] #Used for the output layer of a multilabel network
        self.convolutional = convolutional
        self.flatten_output = True #For convolutional layers
        self.win_size = convolution_win_size #only used for convolutional layers
        self.pooling = np.abs(pooling) #only used for convolutional layers
        self.absolute_pooling = pooling<0 #an option to use absolute-value-max pooling
        self.stride = stride #only used for convolutional layers
        self.padding = padding #Should we pad an image for convolutions?
        self.issparse = False #An option to store the weights in a sparse matrix
        self.levels = []
    
    def copy(self, deep=True):
        """Returns an exact copy of the layer. "deep" also copies over some of the extra parameters
        used for certain analyses after training"""
        new_layer = Layer(self.activation_function)
        new_layer.weights = np.copy(self.weights)
        new_layer.biases = np.copy(self.biases)
        new_layer.mult_factor = self.mult_factor
        new_layer.convolutional = self.convolutional
        new_layer.flatten_output = self.flatten_output
        new_layer.win_size = self.win_size
        new_layer.pooling = self.pooling
        new_layer.absolute_pooling = self.absolute_pooling
        new_layer.stride = self.stride
        new_layer.padding = self.padding
        new_layer.multilabel = self.multilabel
        new_layer.label_lengths = self.label_lengths
        new_layer.levels = self.levels
        new_layer.issparse = self.issparse
        if deep:
            new_layer.subclass_indices = np.copy(self.subclass_indices)
            new_layer.midpoints = np.copy(self.midpoints)
            new_layer.sum_dual_coef = np.copy(self.sum_dual_coef)
            new_layer.support_vectors = self.support_vectors.copy()
        return new_layer
    
    def tanh_to_sigmoid(self):
        
        if np.max(self.levels)>0:
            for w in range(len(self.weights)):
                self.biases[self.levels==w] -= np.array(np.sum(self.weights[w], axis=0)).flatten()
                self.weights[w] *= 2
        else:
            self.biases -= np.array(np.sum(self.weights, axis=0)).flatten()
            self.weights *= 2
        
    
    def sigmoid_to_tanh(self):
        if np.max(self.levels)>0:
            for w in range(len(self.weights)):
                self.biases[self.levels==w] += .5*np.array(np.sum(self.weights[w], axis=0)).flatten()
                self.weights[w] *= .5
        else:
            self.biases += .5*np.array(np.sum(self.weights, axis=0)).flatten()
            self.weights *= .5
                

    def activation(self, nodes):
        """Takes the input to the neurons and runs them through an activation function"""
        if self.symbolic:
            #If the SVM multiplier is large, the outputs are meant to be 0, .5, or 1
            if np.max(np.abs((nodes-np.round(nodes))))<1e-3:
                nodes = np.round(nodes)
            np.putmask(nodes, nodes>0, 1)
            np.putmask(nodes, nodes<0, -1)
            np.putmask(nodes, np.logical_and(nodes>-1, nodes<1), 0)
            nodes += 1
            nodes /= 2
        elif self.activation_function == 'sigmoid':
            nodes = 1/(1+np.exp(-nodes))
        elif self.activation_function == 'tanh':
            nodes = -1 + 2/(1+np.exp(-2*nodes))
        elif self.activation_function == 'softmax':
            nodes -= np.tile(np.max(nodes, axis=1).reshape(-1,1), (1,nodes.shape[1]))
            nodes = np.exp(nodes)
            sum_total = np.sum(nodes, 1)
            nodes = np.divide(nodes, np.tile(sum_total, (nodes.shape[1], 1)).transpose())
        elif self.activation_function =='linear':
            return nodes
        return nodes

    def compute_output(self, input, activate=True, maxpool_return=False, remaining_outputs=None, use_nodes=None, do_pool=True):
        """Returns the output of the layer given the previous layer as input; there is an option to not put
        the weighted sum of inputs through the activation function; also, there is an option to return the
        indices of each of the maxpooling regions when this is used in a convolutional layer"""
        if len(input.shape)==1: #fixes the shape of the input layer to be a 2d array
            input = np.reshape(input,(1, -1))
        if (not len(self.levels)==0) and np.max(self.levels)>0:
            temp_nodes = []
            nodes = []
            previous_inputs = input
            for l in range(np.max(self.levels)+1):
                if not np.any(self.levels==l):
                    temp_nodes.append([])
                    continue
                temp_nodes.append(previous_inputs @ self.weights[l])
                temp_nodes[l] += np.tile(self.biases[self.levels==l], (temp_nodes[l].shape[0], 1))
                temp_nodes[l] = self.activation(temp_nodes[l])
                previous_inputs = np.hstack((previous_inputs, temp_nodes[l]))
                if l==0:
                    nodes = temp_nodes[l]
                else:
                    nodes = np.hstack((nodes, temp_nodes[l]))  
        elif not self.convolutional:
            #Standard method
            if use_nodes is None:
                nodes = input @ self.weights
                if sparse.issparse(nodes):
                    nodes = nodes.toarray()
                nodes = np.add(nodes, np.tile(self.biases, (nodes.shape[0], 1)))
            else: #Only certain nodes are going to be used to compute the output
                nodes = input @ self.weights[:, use_nodes]
                if sparse.issparse(nodes):
                    nodes = nodes.toarray()
                nodes = np.add(nodes, np.tile(self.biases[use_nodes], (nodes.shape[0], 1)))

            if activate:
                nodes = self.activation(nodes)
            if remaining_outputs is not None:
                new_nodes = np.zeros(nodes.shape)
                new_nodes[remaining_outputs] = nodes[remaining_outputs]
                nodes = new_nodes

        else: #Convolutional layer

            #Set parameters
            absolute_pooling = self.absolute_pooling
            if not do_pool:
                pooling = 1
            else:
                pooling = self.pooling
            depth = 1
            if len(input.shape)>2:
                depth = input.shape[2]
            window_indices = Layer.get_subimage_indices(input.shape[1:], self.win_size, self.padding, self.stride, pooling)
            if self.flatten_output:
                nodes = np.zeros((input.shape[0], len(window_indices)*self.num_nodes())) #The output
            else:
                nodes = np.zeros((input.shape[0], len(window_indices), self.num_nodes()))
            if maxpool_return:
                mx_ind = np.zeros(nodes.shape)
            curr_ind = 0

            for i, indices in enumerate(window_indices):
                if absolute_pooling:
                    conv = np.zeros((input.shape[0], self.num_nodes()))
                else:
                    conv = np.full((input.shape[0], self.num_nodes()), -np.inf)
                if maxpool_return:
                    ind = np.zeros(conv.shape)
                for p,p_ind in enumerate(indices):
                    if depth==1:
                        temp_conv = np.matmul(input[:, p_ind], self.weights)
                    else:
                        temp_conv = input[:, p_ind, :].reshape((input.shape[0], self.weights.shape[0])) @ self.weights
                    temp_conv = np.add(temp_conv, np.tile(self.biases, (temp_conv.shape[0], 1)))
                    
                    #Max pooling
                    if not absolute_pooling:
                        conv = np.maximum(conv, temp_conv)
                    else:
                        abs_greater = np.greater(np.abs(conv), np.abs(temp_conv))
                        conv = abs_greater*conv + (1-abs_greater)*temp_conv
                    if maxpool_return:
                        i = np.where(np.equal(conv,temp_conv))
                        ind[i] = p
                    
                if activate: #optional activation
                    conv = self.activation(conv)
                
                #conv = np.full(conv.shape, i)
                
                #Store the maxpooled values in the outputs (nodes)
                if self.flatten_output:
                    nodes[:, curr_ind:curr_ind+self.num_nodes()] = conv
                else:
                    nodes[:,i,:] = conv

                if maxpool_return: #store the indices
                    mx_ind[:, curr_ind:curr_ind+self.num_nodes()] = ind
                curr_ind += self.num_nodes()

            """
            numel = input.shape[1]
            dim = int(np.sqrt(numel))
            depth = 1
            if len(input.shape)>2:
                depth = input.shape[2]
            image_indices = np.reshape(np.arange(numel), (dim, dim))
            pad_border = 0


            if self.padding:
                #Add padded border that extends image outward
                pad_border = int((self.win_size-1)/2)
                image_indices = np.concatenate((image_indices, np.tile(image_indices[:,-1].reshape((-1,1)), (1, pad_border))), axis=1)
                image_indices = np.concatenate((np.tile(image_indices[:,0].reshape((-1,1)), (1, pad_border)), image_indices), axis=1)
                image_indices = np.concatenate((image_indices, np.tile(image_indices[-1,:].reshape((1,-1)), (pad_border, 1))), axis=0)
                image_indices = np.concatenate((np.tile(image_indices[0,:].reshape((1,-1)), (pad_border, 1)), image_indices), axis=0)
                if depth>1:
                    input = np.concatenate((input, np.zeros((input.shape[0], 1, depth))), axis=1)
                else:
                    input = np.concatenate((input, np.zeros((input.shape[0], 1))), axis=1)
                image_indices = image_indices.astype(int)
            
            
            #Get things ready for convolutions
            num_windows_per_sample = len(window_indices)
            nodes = np.zeros((input.shape[0], num_windows_per_sample*self.num_nodes())) #The output
            if maxpool_return:
                mx_ind = np.zeros(nodes.shape)
            curr_ind = 0

            #We will step over all possible windows convolving and taking the max-pooled result
            for x_ind in range(0, dim-self.win_size+1+pad_border*2, pooling + self.stride-1):
                for y_ind in range(0, dim-self.win_size+1+pad_border*2, pooling + self.stride-1):
                    #Initialize output to minus infinity to find the max values
                    if absolute_pooling:
                        conv = np.zeros((input.shape[0], self.num_nodes()))
                    else:
                        conv = np.full((input.shape[0], self.num_nodes()), -np.inf)
                    if maxpool_return:
                        ind = np.zeros(conv.shape)
                    #Step over all the windows in the pooling field
                    for x_pool in range(pooling):
                        for y_pool in range(pooling):
                            #Get the indices from the image that correspond to this window
                            window_indices = image_indices[x_ind+x_pool:x_ind+x_pool+self.win_size, :][:, y_ind+y_pool:y_ind+y_pool+self.win_size].flatten()
                            
                            #Multiply convolution filter and add bias
                            if depth==1:
                                temp_conv = np.matmul(input[:, window_indices], self.weights)
                            else:
                                temp_conv = np.matmul(np.reshape(input[:, window_indices, :], (input.shape[0], self.weights.shape[0])), self.weights)
                            temp_conv = np.add(temp_conv, np.tile(self.biases, (temp_conv.shape[0], 1)))
                            
                            #Max pooling
                            if not absolute_pooling:
                                conv = np.maximum(conv, temp_conv)
                            else:
                                abs_greater = np.greater(np.abs(conv), np.abs(temp_conv))
                                conv = abs_greater*conv + (1-abs_greater)*temp_conv
                            if maxpool_return:
                                i = np.where(np.equal(conv,temp_conv))
                                ind[i] = x_pool*pooling+y_pool

                    if activate: #optional activation
                        conv = self.activation(conv)
                    nodes[:, curr_ind:curr_ind+self.num_nodes()] = conv #store the maxpooled values in the outputs (nodes)
                    if maxpool_return: #store the indices
                        mx_ind[:, curr_ind:curr_ind+self.num_nodes()] = ind
                    curr_ind += self.num_nodes()
            if not self.flatten_output: #Convolutional outputs can be flattened for fully connected layers (i.e. ENN)
                nodes = np.reshape(nodes, (input.shape[0], num_windows_per_sample, self.num_nodes()))
            """
        if maxpool_return:
            return nodes, mx_ind
        else:
            return nodes

    def set_mult_factor(self, new_mult_factor):
        """Set the SVM multiplier for all neurons to a new value and adjust the parameters"""
        if self.symbolic:
            return
        if np.max(self.levels)>0:
            if not isinstance(new_mult_factor, list):
                new_mult_factor = [new_mult_factor for _ in self.weights]
            for w in range(len(self.weights)):
                self.weights[w] *= new_mult_factor[w]/self.mult_factor[w]
                self.biases[self.levels==w] *= new_mult_factor[w]/self.mult_factor[w]
        else:
            self.weights *= new_mult_factor/self.mult_factor
            self.biases *= new_mult_factor/self.mult_factor
        self.mult_factor = new_mult_factor

    def add_node(self, svm, subclass_1=0, subclass_2=0, level=0):
        """Add a node to the layer and store its associated subclasses if it's a differentia neuron"""
        if len(self.biases)==0: #This is the layer's first neuron
            self.weights = svm.weights.reshape(-1, 1)
            self.biases = np.array([svm.bias])
            self.levels = np.array([level])
            if svm.keep_meta_data:
                self.midpoints = svm.midpoint.reshape(-1, 1)
                self.sum_dual_coef = svm.sum_dual_coef
        else: #Add neuron to the others
            if level>0:
                max_level = np.max(self.levels)
                if max_level==0:
                    self.weights = [self.weights]
                    self.mult_factor = [self.mult_factor]
                    if svm.keep_meta_data:
                        self.midpoints = [self.midpoints]
                if level > max_level:
                    for _ in range(level-max_level):
                        self.weights.append(svm.weights.reshape(-1, 1))
                        self.mult_factor.append(1)
                        if svm.keep_meta_data:
                            self.midpoints.append(svm.midpoint.reshape(-1, 1))
                else:
                    self.weights[level] = np.hstack((self.weights[level], svm.weights.reshape(-1, 1)))
                    if svm.keep_meta_data:
                        self.midpoints[level] = np.hstack((self.midpoints[level], svm.midpoint.reshape(-1, 1)))
            elif np.max(self.levels)>0:
                self.weights[0] = np.hstack((self.weights[0], svm.weights.reshape(-1, 1)))
                if svm.keep_meta_data:
                    self.midpoints[0] = np.hstack((self.midpoints[0], svm.midpoint.reshape(-1, 1)))
            else:
                self.weights = np.hstack((self.weights, svm.weights.reshape(-1, 1)))
                if svm.keep_meta_data:
                    self.midpoints = np.hstack((self.midpoints, svm.midpoint.reshape(-1, 1)))
            
            self.biases = np.append(self.biases, svm.bias)
            self.levels = np.append(self.levels, level)
            if svm.keep_meta_data:
                self.sum_dual_coef = np.append(self.sum_dual_coef, svm. sum_dual_coef)

        if max(subclass_1, subclass_2)>=self.subclass_indices.shape[0]: #if the subclasses added are bigger than the current ones
            curr_dim = self.subclass_indices.shape[0]
            num_to_add = max(subclass_1, subclass_2) - curr_dim + 1
            self.subclass_indices = np.concatenate((self.subclass_indices, np.zeros((curr_dim, num_to_add))), axis=1)
            self.subclass_indices = np.concatenate((self.subclass_indices, np.zeros((num_to_add, curr_dim+num_to_add))), axis=0)
        #Place the current neuron in the subclass differentia matrix
        self.subclass_indices[subclass_1, subclass_2] = self.num_nodes()
        if svm.strictness == 0:
            self.subclass_indices[subclass_2, subclass_1] = -self.num_nodes() #Marks that the differentia points the other way
    
    def add_nodes(self, svms, subclasses=None, levels=None):
        """"Add multiple nodes to the layer and store their associated subclasses if differentiae"""
        if levels is None:
            levels = [0 for _ in svms]
        if isinstance(levels, int):
            levels = [levels for _ in svms]
        if self.num_nodes()>0:
            if subclasses is None:
                subclasses = [[0,0] for _ in svms]
            for s_i,s in enumerate(svms):
                self.add_node(s, subclasses[s_i][0], subclasses[s_i][1], levels[s_i])
            return
        
        self.levels = levels

        num_nodes = len(svms)
        num_inputs = np.max(svms[0].weights.shape)
        
        #If a differentia layer, this stores the list of opposing subclasses for each neuron
        if subclasses is not None:
            num_subconcepts = max([max(s[0],s[1]) for s in subclasses])+1
        
        #Extract weights and biases from SVMs and add to the layer
        self.biases = np.zeros(num_nodes)
        if svms[0].issparse: #When we are using sparse matrices, we will store the weights sparsely
            self.issparse = True
            total_nz = 0
            for s in range(num_nodes):
                total_nz += svms[s].weights.nnz
            col_ind = np.zeros(total_nz)
            row_ind = np.zeros(total_nz)
            vals = np.zeros(total_nz)
            count = 0
            for s in range(num_nodes):
                nzs = np.nonzero(svms[s].weights)
                row_ind[count:(count+len(nzs[0]))] = nzs[0]
                col_ind[count:(count+len(nzs[0]))] = np.ones(len(nzs[0]))*s
                vals[count:(count+len(nzs[0]))] = svms[s].weights[nzs]
                self.biases[s] = svms[s].bias
                count += len(nzs[0])
            self.weights = sparse.csc_matrix((vals, (row_ind, col_ind)), shape=(num_inputs, num_nodes))
        else:
            self.weights = np.zeros((num_inputs, num_nodes))
        
            for s in range(num_nodes):
                if svms[s].weights.shape[0]==1:
                    self.weights[:, s] = svms[s].weights[0]
                else:
                    if sparse.issparse(svms[s].weights):
                        self.weights[:, s] = svms[s].weights.toarray().flatten()
                    else:
                        self.weights[:, s] = svms[s].weights.flatten()
                self.biases[s] = svms[s].bias
        
        #If we wish to store some of the meta-data from the SVMs, keep it
        if svms[0].keep_meta_data:
            self.midpoints = np.zeros((num_inputs, num_nodes))
            self.sum_dual_coef = np.zeros(num_nodes)
            for s in range(num_nodes):
                self.midpoints[:, s] = svms[s].midpoint
                self.sum_dual_coef[s] = svms[s].sum_dual_coef
        
        #Store the subclass indices
        if subclasses is not None:
            self.subclass_indices = np.zeros((num_subconcepts, num_subconcepts))
            for s in range(num_nodes):
                self.subclass_indices[subclasses[s][0], subclasses[s][1]] = s+1
                if svms[0].strictness==0:
                    self.subclass_indices[subclasses[s][1], subclasses[s][0]] = -(s+1)

    def num_nodes(self):
        """The number of nodes in the layer"""
        if isinstance(self.weights,list) and len(self.weights)==0:
            return 0
        if len(self.levels)==0 or np.max(self.levels)==0:
            return self.weights.shape[1]
        else:
            return self.biases.size
    
    def num_sv(self):
        """Return the number of support vectors used total in the layer"""
        max_sv = -1
        for i in range(len(self.support_vectors)):
            max_sv = max((max_sv, max(self.support_vectors[i].flatten())))
        ind_as_sv = np.zeros(max_sv+1)
        for i in range(len(self.support_vectors)):
            ind_as_sv[self.support_vectors[i]] = 1
        return sum(ind_as_sv)

    def del_node(self, node_ids):
        """Delete nodes from the layer given node_ids"""
        if self.issparse:
            mask = np.ones(self.weights.shape[1], dtype='bool')
            mask[node_ids] = False
            self.weights = self.weights[:, mask]
        else:
            self.weights = np.delete(self.weights, node_ids, 1)
        self.biases = np.delete(self.biases, node_ids)
        if len(self.sum_dual_coef)>0:
            self.sum_dual_coef = np.delete(self.sum_dual_coef, node_ids)
            self.midpoints = np.delete(self.midpoints, node_ids, 1)
            if len(self.support_vectors)>0:
                self.support_vectors = np.delete(self.support_vectors, node_ids, 0)
        if not type(node_ids)==list:
            node_ids = [node_ids]
        for id in node_ids:
            np.place(self.subclass_indices, np.absolute(self.subclass_indices)==id+1, 0)
        count = 1
        for sc_i in range(self.subclass_indices.shape[0]):
            for sc_j in range(sc_i+1, self.subclass_indices.shape[1]):
                if self.subclass_indices[sc_i, sc_j] > 0:
                    self.subclass_indices[sc_i, sc_j] = count
                    self.subclass_indices[sc_j, sc_i] = -count
                    count += 1
    
    def del_feature(self, feature_ids):
        """Delete features (i.e. input neurons) from the layer given their ids"""
        if self.issparse:
            mask = np.ones(self.weights.shape[0], dtype='bool')
            mask[feature_ids] = False
            self.weights = self.weights[mask, :]
        else:
            self.weights = np.delete(self.weights, feature_ids, 0)
        if len(self.sum_dual_coef)>0:
            self.midpoints = np.delete(self.midpoints, feature_ids, axis=0)

    def reorder_nodes(self, order):
        """Changes the order of the neurons in the layer according to a given order"""
        if len(order) != self.num_nodes():
            return
        self.weights = self.weights[:, order]
        self.biases = self.biases[order]
        if len(self.sum_dual_coef)>0:
            self.sum_dual_coef = self.sum_dual_coef[order]
            self.midpoints = self.midpoints[:, order]
            new_subclass_indices = np.zeros(self.subclass_indices.shape)
            for i in range(self.subclass_indices.shape[0]):
                for j in range(self.subclass_indices.shape[1]):
                    for o_i,o in enumerate(order):
                        if np.absolute(self.subclass_indices[i, j]==o+1):
                            new_subclass_indices[i, j] = (o_i+1)*np.sign(self.subclass_indices[i, j])
                            break
        if len(self.support_vectors)>0:
            self.support_vectors = [self.support_vectors[i] for i in order]

    def reorder_inputs(self, order):
        """When the previous layer is reordered, this reorders the current layer's input features"""
        if len(order) != self.weights.shape[0]:
            return
        self.weights = self.weights[order, :]
        if len(self.midpoints)>0:
            self.midpoints = self.midpoints[order, :]
            
    def get_subclass_diff(self, subclass, both_directions=True):
        """Returns indices of all differentiae associated with the given subclass"""
        sc_ind = [i for i in np.absolute(self.subclass_indices[subclass,:]) if i>0]
        if both_directions:
            sc_ind.extend([i for i in np.absolute(self.subclass_indices[:,subclass]) if i>0])
        return (np.unique(sc_ind)-1).astype(int)
    
    def get_subclass_diff_sign(self, subclass):
        """Returns the sign of the given subclass for all of its associated differentiae"""
        sc_ind = [i for i in range(self.subclass_indices.shape[1]) if np.absolute(self.subclass_indices[subclass,i])>0]
        return np.sign(self.subclass_indices[subclass, sc_ind])
    
    def get_diff_subclasses(self, node):
        """Returns the subclass indices associated with a given differential node"""
        return np.argwhere(self.subclass_indices==node+1)
    
    def set_multilabel(self, label_lengths):
        """Sets the layer to a multilabel output"""
        self.multilabel = True
        self.label_lengths = label_lengths.astype(int)
    
    def rearrange_nodes(self, new_order):
        """Rearranges the node order"""
        if len(new_order) != self.num_nodes():
            return False
        self.weights = self.weights[:, new_order]
        self.biases = self.biases[new_order]
    
    def rearrange_features(self, new_order):
        """Rearranges the feature order"""
        if len(new_order) != self.weights.shape[0]:
            return False
        self.weights = self.weights[new_order, :]
    
    def desparsify(self):
        #Make everything non-sparse
        if sparse.issparse(self.weights):
            self.issparse = False
            self.weights = self.weights.toarray()
            if self.midpoints is not None:
                self.midpoints = self.midpoints.toarray()
    
    def save(self, filename):
        np.savez(filename, weights=self.weights, biases=self.biases, levels=self.levels,
                activation=self.activation, subclass_indices=self.subclass_indices)
    
    @staticmethod
    def load(filename):
        layer_file = np.load(filename, allow_pickle=True)
        layer = Layer(layer_file['activation'])
        layer.weights = layer_file['weights']
        layer.biases = layer_file['biases']
        layer.levels = layer_file['levels']
        layer.subclass_indices = layer_file['subclass_indices']
        return layer

    
    @staticmethod
    def get_subimage_indices(shape, win_size, padding=False, stride=1, pooling=1):

        numel = shape[0]
        dim = int(np.sqrt(numel))
        image_indices = np.reshape(np.arange(numel), (dim, dim))
        pad_border = 0

        if padding:
            #Add padded border that extends image outward
            pad_border = int((win_size-1)/2)
            image_indices = np.concatenate((image_indices, np.tile(image_indices[:,-1].reshape((-1,1)), (1, pad_border))), axis=1)
            image_indices = np.concatenate((np.tile(image_indices[:,0].reshape((-1,1)), (1, pad_border)), image_indices), axis=1)
            image_indices = np.concatenate((image_indices, np.tile(image_indices[-1,:].reshape((1,-1)), (pad_border, 1))), axis=0)
            image_indices = np.concatenate((np.tile(image_indices[0,:].reshape((1,-1)), (pad_border, 1)), image_indices), axis=0)
            image_indices = image_indices.astype(int)
        
        #Get things ready for convolutions
        indices = []

        #We will step over all possible windows convolving and taking the max-pooled result
        for x_ind in range(0, dim-win_size+1+pad_border*2, pooling + stride - 1):
            for y_ind in range(0, dim-win_size+1+pad_border*2, pooling + stride - 1):
                #Step over all the windows in the pooling field
                window_indices = []
                for x_pool in range(pooling):
                    for y_pool in range(pooling):
                        #Get the indices from the image that correspond to this window
                        window_indices.append(image_indices[x_ind+x_pool:x_ind+x_pool+win_size, :][:, y_ind+y_pool:y_ind+y_pool+win_size].flatten())
                indices.append(window_indices)
        
        return indices