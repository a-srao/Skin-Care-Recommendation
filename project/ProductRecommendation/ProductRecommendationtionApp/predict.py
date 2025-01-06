
import numpy as np 
import cv2
import os
import re
import pandas as pd
from os import getcwd
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.feature import greycomatrix, greycoprops
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sb

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

dataset=pd.read_csv('featureExtracted-skin.csv')

def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
        
        glcm = greycomatrix(img, 
                            distances=dists, 
                            angles=agls, 
                            levels=lvl,
                            symmetric=sym, 
                            normed=norm)
        
        feature = []
        glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
        for item in glcm_props:
                feature.append(item)
        # feature.append(label) 
        
        return feature


def skin_classify(test_glcm_features):
    X=np.array(dataset.iloc[:,1:-1])
    X=X.astype(dtype='int')
    Y=np.array(dataset.iloc[:,-1])
    Y=Y.reshape(-1,)

    X_train, X_test, y_train, y_test =train_test_split(X,Y,test_size=0.25,
                                                    random_state=42)

    print(X_train.shape)
    print(type(X_test))
    model_RR=RandomForestClassifier(n_estimators=100,criterion='entropy',)
    model_RR.fit(X_train,y_train)

    y_predicted_RR=model_RR.predict(np.asarray(test_glcm_features).reshape(1,-1))
    return y_predicted_RR
       
   
def process():
    
    image=cv2.imread(getcwd() + "\\media\\uploaded_image\\input.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ymin, ymax, xmin, xmax = h//3, h*2//3, w//3, w*2//3
    crop = gray[ymin:ymax, xmin:xmax]
            
    resize = cv2.resize(crop, (0,0), fx=0.5, fy=0.5)

    

    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    glcm_all_agls = []
    columns = []
    angles = ['0', '45', '90','135']
    glcm_features=calc_glcm_all_agls(resize, None, props=properties)
    print(glcm_features)
    segments = slic(image, n_segments=100, compactness=10, sigma=1)
    index=skin_classify(glcm_features)
    G = nx.Graph()


    for segment_id in np.unique(segments):
        segment_pixels = np.where(segments == segment_id)
        node_features = np.mean(image[segment_pixels], axis=0)  # Example: using mean color as node feature
        G.add_node(segment_id, features=node_features)

        # Add edges between neighboring segments
        for i in range(image.shape[0] - 1):
                for j in range(image.shape[1] - 1):
                        segment_id1 = segments[i, j]
                        segment_id2 = segments[i + 1, j]
                        if segment_id1 != segment_id2:
                                G.add_edge(segment_id1, segment_id2)

                                segment_id2 = segments[i, j + 1]
                        if segment_id1 != segment_id2:
                                G.add_edge(segment_id1, segment_id2)

        # Example of accessing node features
    node_id = 0
    # print("Node Features:", G.nodes[node_id]['features'])
    print("Node Features:", G.nodes)

# Example of accessing edge information
    edge = (0, 1)  # Example edge between node 0 and node 1
    print("Edge Weight:", G.edges)



    Data(x=G.nodes) 
    model=torch.load('trained_model.pt')

    predict=model.eval()


    
    print("Result: ",index)
    return index[0]




