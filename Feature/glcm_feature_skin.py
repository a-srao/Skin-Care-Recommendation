import numpy as np 
import cv2
import os
import re
import pandas as pd
from skimage.feature import greycomatrix, greycoprops

path='skindataset/'


images=[]
labels=[]

for folder in os.listdir(path):  
    for img in os.listdir(path+folder):
        print('Reading image from :' , path+folder+'/'+img)
        image=cv2.imread(path+folder+'/'+img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        ymin, ymax, xmin, xmax = h//3, h*2//3, w//3, w*2//3
        crop = gray[ymin:ymax, xmin:xmax]
                
        resize = cv2.resize(crop, (0,0), fx=0.5, fy=0.5)
        images.append(resize)
        labels.append(folder)

properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
glcm_all_agls = []
columns = []
angles = ['0', '45', '90','135']

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
    feature.append(label) 
    
    return feature

for img, label in zip(images, labels): 
    glcm_all_agls.append(
            calc_glcm_all_agls(img, 
                                label, 
                                props=properties)
    )


for name in properties:
    for ang in angles:
        columns.append(name + "_" + ang)

columns.append("label")
glcm_df = pd.DataFrame(glcm_all_agls, 
                      columns = columns)

glcm_df.head(15)

glcm_df.to_csv('featureExtracted-skin.csv',encoding='utf-8')