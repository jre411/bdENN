"""Trains a linear SVM"""

#Imports
import numpy as np
import scipy.sparse as sparse
from sklearn import svm
from sklearn import base
import warnings
#import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sklearn.linear_model import LogisticRegression

class SVM():
    """An SVM object"""
    #This class exists to store and modify the results of the SVMS used in training ENNs

    def __init__(self, input_1, input_2, svm_cost=1, fast=False, symbolic=False, store_duals=False, strictness=0, logit=False, exemplar=False, prototype=False):
        self.y_train = [] # all class values, set as 1 or -1
        self.x_train = [] # all data points
        self.symbolic = symbolic
        self.issparse = False
        self.store_duals = store_duals
        self.duals = []
        self.margin = 0
        if sparse.issparse(input_1):
            self.issparse = True
            self.x_train = sparse.vstack([input_1, input_2]).tocsc()
            self.y_train = np.concatenate((np.ones((input_1.shape[0],)), -np.ones((input_2.shape[0],))))
        elif input_2.ndim > 1: # handles the case when the two inputs are data points from different classes
            self.x_train = np.concatenate((input_1, input_2), axis=0)
            self.y_train = np.concatenate((np.ones((input_1.shape[0],)), -np.ones((input_2.shape[0],))))
        else: # handles the case when the two inputs are already X and y
            self.x_train = input_1
            self.y_train = input_2

        self.weights = np.zeros((self.x_train.shape[1],)) # the weights assigned to each feature
        self.bias = 0 # bias value of the svm
        self.strictness = strictness
        self.features = [i for i in range(self.x_train.shape[1])] # features to use in training
        self.sum_dual_coef = 0 # this information is necessary when reconstructing nearest point on convex hull
        self.fast = fast #if fast, the LinearSVC class is used instead and with looser parameters
        #this is for fast training, but less information stored does not permit certain analyses
        self.logit = logit
        if sum(self.y_train>0)<5:
            self.logit = False
        self.keep_meta_data = not (self.fast or self.issparse or self.logit)
        if self.logit:
            self.y_train = (self.y_train>0).astype(int)
        if self.keep_meta_data:
            self.midpoint = np.zeros((self.x_train.shape[1],)) # the intersection point of the SVM
            self.support_vector_indices = np.array([]) # the indices of the SVM's support vectors
        else:
            self.midpoint = None
            self.support_vector_indices = None
        self.use_exact = (self.x_train.shape[0]==2) #When only two points are being separating, can get exact result
        if not self.use_exact:
            self.use_exact_2 = False#np.sum(self.y_train>0)==1
        self.id = None
        self.exemplar = exemplar
        self.prototype = prototype
        
        if not self.use_exact:
            if fast:
                if self.logit:
                    self.svc = LogisticRegression()
                elif fast>1:
                    if True:
                        self.svc = svm.SVC(kernel='linear', C=svm_cost, class_weight='balanced', max_iter=10000, tol=1e-5)
                    else:
                        self.svc = svm.LinearSVC(C=svm_cost, class_weight='balanced', max_iter=500, tol=1e-5)
                else:
                    if True:
                        self.svc = svm.SVC(kernel='linear', C=svm_cost, class_weight='balanced', max_iter=10000, tol=1e-5)
                    else:
                        self.svc = svm.LinearSVC(C=svm_cost, class_weight='balanced')
            else:
                if self.logit:
                    self.svc = LogisticRegression()
                self.svc = svm.SVC(kernel='linear', C=svm_cost, class_weight='balanced', max_iter=50000, tol=1e-30)
                
    
    def copy(self):
        #Creates a copy of the SVM
        new_svm = SVM(self.x_train, self.y_train, self.svc.C, fast=self.fast, symbolic=self.symbolic)
        if self.keep_meta_data:
            new_svm.midpoint = np.copy(self.midpoint)
            new_svm.support_vector_indices = np.copy(self.support_vector_indices)
        new_svm.weights = np.copy(self.weights)
        new_svm.bias = np.copy(self.bias)
        new_svm.features = np.copy(self.features)
        new_svm.issparse = self.issparse
        new_svm.keep_meta_data = self.keep_meta_data
        new_svm.use_exact = self.use_exact
        new_svm.use_exact_2 = self.use_exact_2
        new_svm.store_duals = self.store_duals
        new_svm.duals = self.duals.copy()
        new_svm.id = self.id
        new_svm.strictnes = self.strictness
        return new_svm
    
    def flip_direction(self):
        """Flips the direction the SVM points"""
        self.weights *= -1
        self.bias *= -1
        self.strictness *= -1

        return self


    def trim(self):
        """In order to save on memory, one can remove unecessary information from the SVM"""
        self.x_train = None
        self.y_train = None
        self.svc = None
        if not self.keep_meta_data:
            self.features = None
            self.support_vector_indices = None
            self.midpoint = None
    
    def change_strictness(self, strictness=None):
        #Change the strictness of the SVM; the strictness says how much to change the bias factor
        #such that it moves the hyperplane more toward either class
        if strictness is None:
            strictness = self.strictness
        outputs1 = self.x_train[[i for i in range(self.x_train.shape[0]) if self.y_train[i]>0],:] @ self.weights
        outputs2 = self.x_train[[i for i in range(self.x_train.shape[0]) if self.y_train[i]<=0],:] @ self.weights
        min_1 = np.min(outputs1)
        max_2 = np.max(outputs2)
        self.bias = -max_2 - (min_1-max_2)*(strictness*.999+1)/2
        if self.symbolic:
            self.symbolize_weights()
    
    def train(self, dual_coeff=None):
        """Train the SVM"""
        if self.prototype:
            prototype1 = np.mean(self.x_train[self.y_train>0, :][:, self.features], axis=0)
            prototype2 = np.mean(self.x_train[self.y_train<=0, :][:, self.features], axis=0)
            new_weights = (prototype1 - prototype2).transpose()
            self.sum_dual_coef = 1
            midpoint = (prototype1 + prototype2)/2
            self.bias = -np.sum(np.multiply(new_weights, midpoint))
            if self.keep_meta_data:
                self.support_vector_indices = np.arange(2)
                new_midpoint = (prototype1 + prototype2)/2
                if self.issparse:
                    self.midpoint = new_midpoint[0, self.features].transpose()
                else:
                    self.midpoint = new_midpoint[self.features]
            if self.store_duals:
                self.duals = np.array([1,-1]).reshape((1,-1))
            
            #Store the margin of the SVM
            factor = np.sum(prototype1*new_weights) + self.bias
            if self.issparse:
                self.margin = 2/np.sqrt(np.sum(np.square(new_weights.data/factor)))
            else:
                self.margin = 2/np.sqrt(np.sum(np.square(new_weights/factor)))
        if self.exemplar:
            ind1 = [i for i,y in enumerate(self.y_train) if y>0]
            ind2 = [i for i,y in enumerate(self.y_train) if y<=0]
            mags1 = np.sum(np.abs(self.x_train[ind1,:]), axis=1)
            mags2 = np.sum(np.abs(self.x_train[ind2,:]), axis=1)
            all_mags = mags1.reshape(1,-1) + mags2.reshape(-1,1)

            order = np.argsort(all_mags.flatten())

            found = False
            for o in order:
                i1 = ind1[int(o/len(ind2))]
                i2 = ind2[int(o%len(ind2))]

                new_weights = (self.x_train[i1, self.features] - self.x_train[i2, self.features]).reshape(-1,1)
                dist1 = self.x_train[ind1, :][:, self.features] @ new_weights
                dist2 = self.x_train[ind2, :][:, self.features] @ new_weights
                min_1 = np.min(dist1)
                max_2 = np.max(dist2)
                if min_1>max_2:
                    found = True
                    self.bias = -(min_1+max_2)/2
                    midpoint = (self.x_train[i1, self.features] + self.x_train[i2, self.features])/2
                    factor = np.sum(self.x_train[0, self.features]*new_weights) + self.bias
                    if self.issparse:
                        self.margin = 2/np.sqrt(np.sum(np.square(new_weights.data/factor)))
                    else:
                        self.margin = 2/np.sqrt(np.sum(np.square(new_weights/factor)))
                    self.keep_meta_data = False
                    break
            if not found:
                debug=1
        elif dual_coeff is None:
            if self.use_exact:
                #This method directly draws a hyperplane between the two opposing points
                #This is necessary to avoid small errors in the SVM when exact results are needed
                new_weights = (self.x_train[0, self.features] - self.x_train[1, self.features]).transpose()
                self.sum_dual_coef = 1
                midpoint = (self.x_train[0, self.features] + self.x_train[1, self.features])/2
                self.bias = -np.sum(np.multiply(new_weights, midpoint))
                if self.keep_meta_data:
                    self.support_vector_indices = np.arange(2)
                    new_midpoint = (self.x_train[0, self.features] + self.x_train[1, self.features])/2
                    if self.issparse:
                        self.midpoint = new_midpoint[0, self.features].transpose()
                    else:
                        self.midpoint = new_midpoint[self.features]
                if self.store_duals:
                    self.duals = np.array([1,-1]).reshape((1,-1))
                
                #Store the margin of the SVM
                factor = np.sum(self.x_train[0, self.features]*new_weights) + self.bias
                if self.issparse:
                    self.margin = 2/np.sqrt(np.sum(np.square(new_weights.data/factor)))
                else:
                    self.margin = 2/np.sqrt(np.sum(np.square(new_weights/factor)))
                
            else:
                if self.use_exact_2>0:
                    #An alternative, optional method for quickly getting a rough SVM by comparing each side's mean
                    minus_ind = np.where(self.y_train<=0)[0]
                    pos_ind = np.where(self.y_train>0)[0]
                    remain_ind = minus_ind.copy()
                    for _ in range(1000):
                        mn = np.mean(self.x_train[remain_ind,:][:,self.features], axis=0)
                        h = (self.x_train[pos_ind,self.features] - mn).transpose()
                        midpoint = (self.x_train[pos_ind,self.features] + mn)/2
                        b = -np.sum(midpoint*h)
                        dist = self.x_train[minus_ind,:][:, self.features] @ h
                        mx = np.max(dist)
                        new_ind = [minus_ind[i] for i,d in enumerate(dist) if d==mx]
                        if len(new_ind)>=len(remain_ind):
                            break
                        remain_ind = new_ind
                    new_weights = h
                    self.bias = b
                    factor = np.sum(self.x_train[0, self.features]*new_weights) + self.bias
                    if self.issparse:
                        self.margin = 2/np.sqrt(np.sum(np.square(new_weights.data/factor)))
                    else:
                        self.margin = 2/np.sqrt(np.sum(np.square(new_weights/factor)))
                if self.use_exact_2==0:
                    #Train using an actual SVM and store data
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        self.svc.fit(self.x_train[:, self.features], self.y_train)
                    if self.keep_meta_data:
                        #Without the LinearSVC we can also collect some of this information directly for later use
                        self.support_vector_indices = self.svc.support_
                        new_midpoint = np.matmul(self.svc.support_vectors_.transpose(), np.absolute(self.svc.dual_coef_.transpose())) \
                            / np.sum(np.absolute(self.svc.dual_coef_))
                        self.midpoint = np.zeros(self.x_train.shape[1])
                        for i,feat in enumerate(self.features):
                            self.midpoint[feat] = new_midpoint[i]
                        self.sum_dual_coef = np.sum(np.absolute(self.svc.dual_coef_))
                        if self.issparse:
                            self.midpoint = sparse.csc_matrix(self.midpoint)
                    self.bias = self.svc.intercept_
                    new_weights = self.svc.coef_.transpose()
                    if self.store_duals:
                        self.duals = np.zeros((1, self.x_train.shape[0]))
                        self.duals[0, self.svc.support_] = np.absolute(self.svc.dual_coef_) * self.y_train[self.svc.support_]
                    if self.issparse:
                        self.margin = 2/np.sqrt(np.sum(np.square(new_weights.data)))
                    else:
                        self.margin = 2/np.sqrt(np.sum(np.square(new_weights)))
        else: #reuse dual coefficients
            new_weights = (dual_coeff @ self.x_train[:, self.features]).transpose()
            outputs = self.x_train[:, self.features] @ new_weights
            min_1 = np.min(outputs[self.y_train>0])
            max_2 = np.max(outputs[self.y_train<=0])
            self.bias = -(min_1+max_2)/2
            self.margin = 2/np.sqrt(np.sum(np.square(new_weights)))
        
        #Save the weights with all possible features, leaving as zero those features that were ignored.
        if self.x_train.shape[1]==len(self.features):
            self.weights = new_weights.copy()
        else:
            if self.issparse:
                self.weights = sparse.csc_matrix((self.x_train.shape[1], 1))
                self.weights[self.features] = new_weights
            else:
                self.weights = np.zeros(self.x_train.shape[1])
                self.weights[self.features] = new_weights[:, 0]
        
        #If we want the network to do symbolic manipulation, it can make the weights more symbolic
        #This means standardizing weights and turning them into integers
        if self.symbolic:
            self.symbolize_weights()
        elif not self.use_exact:
            self.equalize_weights() #This takes weights that are very similar and makes them the same
        
        #Deal with sparse inputs and weights
        if self.issparse:
            if not sparse.issparse(self.weights):
                self.weights = sparse.csc_matrix(self.weights)
            else:
                self.weights = self.weights.tocsc()
            if self.midpoint is not None:
                self.midpoint = sparse.csc_matrix(self.midpoint)
        
        #Change the placement of the SVM hyperplane by modifying the bias term
        if self.strictness != 0:
            self.change_strictness()

    def blur(self):
        """This is an optional way to add a gaussian blur to the weights to see if that works better"""
        if self.weights.shape[0] == 784:
            self.weights = gaussian_filter(np.reshape(self.weights, (28, 28)), 1, mode='constant', cval=0)
            self.weights.flatten()
        
    def equalize_weights(self):
        """If weights are very close, set them equal to each other. This is done to ensure that for more exact tasks
        we have weights with more exact values"""
        if self.symbolic:
            self.symbolize_weights()
            return

        #Find weights within a given tolerance and set them all to the same value
        mx_value = np.max(np.absolute(self.weights))
        if not self.issparse:
            already_done = np.zeros(self.weights.shape[0])
            for i in range(len(already_done)):
                if already_done[i]:
                    continue
                close_ind = np.where(np.isclose(np.absolute(self.weights[i:]), np.absolute(self.weights[i]), atol=.0001*mx_value))[0]
                new_mean = np.max(np.absolute(self.weights[i+close_ind]))
                already_done[i+close_ind] = True
                self.weights[i+close_ind] = new_mean*np.sign(self.weights[i+close_ind])
        else:
            nz_ind = np.nonzero(self.weights)[0]
            new_weights = self.weights[nz_ind].toarray()
            already_done = np.zeros(new_weights.shape[0])
            for i in range(len(already_done)):
                if already_done[i]:
                    continue
                close_ind = np.where(np.isclose(np.absolute(new_weights[i:]), np.absolute(new_weights[i]), atol=.0001*mx_value))[0]
                new_mean = np.max(np.absolute(new_weights[i+close_ind]))
                already_done[i+close_ind] = True
                new_weights[i+close_ind] = new_mean*np.sign(new_weights[i+close_ind])
            self.weights[nz_ind] = new_weights

    def symbolize_weights(self):
        """Sets all of the weights to integer values"""
        
        #All of the values
        factor = np.max(np.abs(self.weights))
        if factor != 0:
            self.weights = self.weights.astype(float) / factor
            self.bias = float(self.bias) / factor
        self.weights[np.abs(self.weights)<.2] = 0 
        #self.weights[np.abs(self.weights)<1e-5] = 0 
        w = np.absolute(np.squeeze(np.array(self.weights[self.weights!=0])))
        if np.abs(self.bias)>1e-4:
            w = np.hstack((w, np.absolute(self.bias)))
        if isinstance(w, int):
            w = np.array(w)
        if len(w)==0:
            if np.abs(self.bias)>.02:
                self.bias = np.sign(self.bias)
            return
        mn_value = np.min(w)
        best_value = 1
        best_diff = np.inf
        #We are going to divide everything by the minimum value; if everything is not an integer,
        #we will try scaling it up (for example, if weights were 2 and 2.5 we would want to scale
        #them up to 4 and 5)
        for i in range(10):
            temp_w = w/mn_value*(i+1)
            mx_diff = np.max(np.absolute(temp_w - np.round(temp_w)))
            if mx_diff<=.025:
                best_value = i+1
                break
            elif mx_diff<best_diff:
                best_diff = mx_diff
                best_value = i+1
        self.weights = np.round(self.weights/mn_value*best_value)
        self.bias = np.round(self.bias/mn_value*best_value)
        return
        all_w = np.unique(np.abs(self.weights[self.weights!=0]))
        changed = False
        for i,w in enumerate(all_w[:-1]):
            new_weights = self.weights.copy()
            ind = np.where(np.abs(new_weights)==w)[0]
            new_weights[ind] = all_w[i+1]*np.sign(self.weights[ind])
            new_labels = (self.x_train @ new_weights + self.bias).flatten()
            if np.all(np.sign(new_labels)==self.y_train):
                self.weights = new_weights
                dist1 = np.min(new_labels[self.y_train>0])
                dist0 = np.max(new_labels[self.y_train<=0])
                self.bias += -(dist1+dist0)/2
                changed = True
        
        if changed:
            self.symbolize_weights()

        
    def get_margin(self):
        """Compute the margin of the SVM"""
        return self.margin

    def set_features(self, ind):
        """Set the features to use in training"""
        self.features = ind
    
    def get_misclass_error(self):
        """Compute the SVM's misclassification error"""
        if not self.issparse:
            outputs1 = np.matmul(self.x_train[[i for i in range(self.x_train.shape[0]) if self.y_train[i]>0],:], self.weights) + self.bias
            outputs2 = np.matmul(self.x_train[[i for i in range(self.x_train.shape[0]) if self.y_train[i]<=0],:], self.weights) + self.bias
        else:
            outputs1 = (self.x_train[[i for i in range(self.x_train.shape[0]) if self.y_train[i]>0],:] @ self.weights).toarray() + self.bias
            outputs2 = (self.x_train[[i for i in range(self.x_train.shape[0]) if self.y_train[i]<=0],:] @ self.weights).toarray() + self.bias
        err1 = np.mean(outputs1<0)
        err2 = np.mean(outputs2>0)
        return (err1+err2)/2

    def de_featurize(self, total_features, feature_ind):
        """Set the old features among other zero features"""
        if self.keep_meta_data:
            new_midpoint = np.zeros((total_features,))
            for i in range(len(feature_ind)):
                new_midpoint[feature_ind[i]] = self.midpoint[i]
            self.midpoint = new_midpoint
        
        new_weights = np.zeros((total_features,))
        for i in range(len(feature_ind)):
            new_weights[feature_ind[i]] = self.weights[i]
        if self.issparse:
            self.weights = sparse.csc_matrix(new_weights.reshape(-1, 1))
        else:
            self.weights = np.array(new_weights)
        self.features = feature_ind
        