'''
Created on Feb 22, 2019

@author: daniel
'''
from MorphologicalSkeletonTransform import MorphologicalSkeletonTransform
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    B = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    plt.imshow(B)
    plt.gray()
    plt.title("Structuring Element")
    plt.show()
    mst = MorphologicalSkeletonTransform()
    if len(sys.argv[1:]) == 0:
        sys.argv[1:] = ["Data/deer.png", "Data/dog.png", "Data/butterfly.png", "Data/lamp.png", "Data/fish.png"]
    

    imgs = []
    for arg in sys.argv[1:]:
        imgs.append(cv2.imread(arg))
    for X in imgs:
        
        ## preprocessing step...
        X = np.bitwise_not(X)
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        _,X = cv2.threshold(X, 0, 255, cv2.THRESH_OTSU)
        #X = X >= threshold_global_otsu
        
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