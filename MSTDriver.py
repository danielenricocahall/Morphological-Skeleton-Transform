'''
Created on Feb 22, 2019

@author: daniel
'''
from MorphologicalSkeletonTransform import MorphologicalSkeletonTransform
import sys
import cv2
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

def main():
    B = np.zeros((4,4), dtype = np.uint8)
    B[0,1:3] = 1
    B[1,:] = 1
    B[2,:] = 1
    B[3, 1:3] = 1

    
    plt.imshow(B)
    plt.gray()
    plt.title("Structuring Element")
    plt.show()
    mst = MorphologicalSkeletonTransform()
    if len(sys.argv[1:]) == 0:
        sys.argv[1:] = ["Data/deer.png", "Data/dog.png", "Data/butterfly.png", "Data/lamp.png", "Data/fish.png"]
    

    imgs = []
    for arg in sys.argv[1:]:
        print(arg)
        imgs.append(cv2.imread(arg))
    for X in imgs:
        
        ## preprocessing step...
        X = np.bitwise_not(X)
        X = rgb2gray(X)
        threshold_global_otsu = threshold_otsu(X)
        X = X >= threshold_global_otsu
        
        S = mst.computeSkeletonSubsets(X, B)
        
        fig = plt.figure()
        plt.gray()
        fig.add_subplot(1, len(S) + 1, 1)
        plt.imshow(X)
        plt.title("Original Image")
        plt.axis('off')
        for i,s in enumerate(S):
            fig.add_subplot(1,len(S)+1,i+2)
            plt.imshow(s)
            plt.title("$S_{" + str(i) + "}$")
            plt.axis('off')
        plt.show()
        
        (components, reconstructions) = mst.reconstructImage(S, X, B)
        
                        
        fig = plt.figure()
        plt.gray()
        fig.add_subplot(1, len(components) + 1, 1)
        plt.imshow(X)
        plt.title("Original Image")
        plt.axis('off')
            
        for n,component in enumerate(components):
            fig.add_subplot(1,len(components)+1,n+2)
            title = "$S_{" + str(n) + "}"

            if n > 1:
                    title = title + "\oplus " + str(n) + "B"
            else:
                title = title + "\oplus B"
            title = title + "$"
            plt.imshow(component)
            plt.title(title)
            plt.axis('off')
        plt.show()
        
        fig = plt.figure()
        fig.add_subplot(121)
        plt.imshow(X)
        plt.title('Original Image')
        plt.axis('off')
        fig.add_subplot(122)
        plt.imshow(reconstructions[-1])
        plt.title('Reconstruction')
        plt.axis('off')
        plt.show()

            
            
    
    
    
if __name__ == "__main__":
    main()
    exit()