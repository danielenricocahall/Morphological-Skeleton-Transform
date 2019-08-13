'''
Created on Feb 22, 2019

@author: daniel
'''

import numpy as np
import cv2

class MorphologicalSkeletonTransform():
        
    def computeSkeletonSubsets(self, X, B):
        S = []
        X_n = X 
        while np.count_nonzero(X_n) > 0:
            X_prev = X_n
            X_n = cv2.erode(X_n, B)
            S_n = X_prev - cv2.dilate(X_n, B)
            S.append(S_n)
        return S
    
    def reconstructImage(self, S, X, B):
        X_reconstructed = np.zeros_like(X, dtype = np.uint8)
        reconstructions = []
        components = []
        for n,s in enumerate(S):
            component = cv2.dilate(s, B, iterations = n)
            X_reconstructed = np.logical_or(X_reconstructed, component)
            components.append(component)
            reconstructions.append(X_reconstructed)
        return (components, reconstructions)
    
    