#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:48:02 2023

@author: ajitj
"""

from skimage.io import imread
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import pandas as pd
import utils, defaults
import seaborn as sns

pars = defaults.DefaultPars()
masks = imread('../20221006_red_uncalibrated_D12_1_Simple Segmentation.tiff')
red   = imread('../20221006_red_uncalibrated_D12_1.tif')

def meanint(label, image):
    
    mult = label * image
    
    return mult[np.nonzero(mult)].mean()

nuclei_area = []
plates_area = []
nuclei_maax = []
plates_maax = []
nuclei_miax = []
plates_miax = []
nuclei_extnt = []
plates_extnt = []
nuclei_peri = []
plates_peri = []
nuclei_mean  = []
plates_mean  = []

for plane in np.arange(masks.shape[0]):
    im = masks[plane, :, :]
    r  = red[plane, :, :]
    nuclei = im == 3
    plates = im == 1
    
    nuclei_props = regionprops(label(nuclei), intensity_image=r, 
                               extra_properties=[meanint])
    plates_props = regionprops(label(plates), intensity_image=r,
                               extra_properties=[meanint])
    
    
    for nucleus in np.arange(len(nuclei_props)):
        nuclei_area.append(nuclei_props[nucleus].area)
        nuclei_maax.append(nuclei_props[nucleus].axis_major_length)
        nuclei_miax.append(nuclei_props[nucleus].axis_minor_length)
        nuclei_extnt.append(nuclei_props[nucleus].extent)
        nuclei_peri.append(nuclei_props[nucleus].perimeter)
        nuclei_mean.append(nuclei_props[nucleus].meanint)
        
    for plate in np.arange(len(plates_props)):
        plates_area.append(plates_props[plate].area)
        plates_maax.append(plates_props[plate].axis_major_length)
        plates_miax.append(plates_props[plate].axis_minor_length)
        plates_extnt.append(plates_props[plate].extent)
        plates_peri.append(plates_props[plate].perimeter)
        plates_mean.append(plates_props[plate].meanint)
    
    print(plane)


nuclei_props = pd.DataFrame({ 
                             'area':        nuclei_area,
                             'major':       nuclei_maax,
                             'minor':       nuclei_miax,
                             'extent':      nuclei_extnt,
                             'perimeter':   nuclei_peri,
                             'meanint':     nuclei_mean})

plates_props = pd.DataFrame({ 
                             'area':        plates_area,
                             'major':       plates_maax,
                             'minor':       plates_miax,
                             'extent':      plates_extnt,
                             'perimeter':   plates_peri,
                             'meanint':     plates_mean
                             })


n = nuclei_props.loc[(nuclei_props.area > 500)]\
                .loc[ (nuclei_props.area < pars.nucsize_max)]
                
p = plates_props.loc[(plates_props.area > 500)]\
                .loc[ (plates_props.area < pars.nucsize_max)]

n['label'] = 0
p['label'] = 1

combined = pd.concat([n,p])
Y = combined.label
#%% Filtering and Plotting
n1 = nuclei_props.loc[(nuclei_props.area > 500)]\
                .loc[ (nuclei_props.area < pars.nucsize_max)]
                
p1 = plates_props.loc[(plates_props.area > 500)]\
                .loc[ (plates_props.area < pars.nucsize_max)]
                
n1['label'] = 0
p1['label'] = 1
c1 = pd.concat([n1,p1])

## Need to normalize mean intensity to avoid effects of differential labeling


#%%
# Trim the highly correlated properties before training
combined.drop(columns=['label'], inplace = True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
                                                    combined, Y, 
                                                    test_size=0.33, 
                                                    random_state=42
                                                    )
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

#%% Success rate

success = y_test.to_numpy() - preds
print(len(success[np.nonzero(success)]))

#%% Save model
import pickle

# save the model to disk
filename = 'meta_plate_normalized.model'
pickle.dump(clf, open(filename, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)

# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# kernel = 1.0 * RBF(1.0)
# gpc = GaussianProcessClassifier(kernel=kernel,
#                                 random_state=0).fit(X_train, y_train)
# print(gpc.score(X_train, y_train))
# probs = gpc.predict_proba(X_test)
