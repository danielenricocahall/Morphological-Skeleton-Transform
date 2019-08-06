'''
Created on Feb 22, 2019

@author: daniel
'''

import numpy as np
from skimage.morphology import binary_dilation, binary_erosion

class MorphologicalSkeletonTransform():
        
    def computeSkeletonSubsets(self, X, B):
        S = []
        X_n = X       
        while np.count_nonzero(X_n) > 0:
            X_prev = X_n
            X_n = binary_erosion(X_n, B)
            S_n = np.logical_and(X_prev, np.logical_not(binary_dilation(X_n, B)))
            S.append(S_n)
        return S
    
    def reconstructImage(self, S, X, B):
        w, h = X.shape[0], X.shape[1]
        X_reconstructed = np.zeros((w, h), dtype = np.uint8)
        reconstructions = []
        components = []
        
        
        for n,s in enumerate(S):
            if n > 0:
                component = s
                for _ in range(n):
                    component = binary_dilation(component,B)
                X_reconstructed = np.logical_or(X_reconstructed, component)
            else:
                X_reconstructed = np.logical_or(X_reconstructed, s)
                component = s
            components.append(component)
            reconstructions.append(X_reconstructed)
        return (components, reconstructions)
    
    