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
masks = imread('20221006_red_uncalibrated_D12_1_Simple Segmentation.tiff')

nuclei_area = []
plates_area = []
nuclei_ecce = []
plates_ecce = []
nuclei_maax = []
plates_maax = []
nuclei_miax = []
plates_miax = []
nuclei_extnt = []
plates_extnt = []
nuclei_peri = []
plates_peri = []
nuclei_carea = []
plates_carea = []
nuclei_feret = []
plates_feret = []

for plane in np.arange(masks.shape[0]):
    im = masks[plane, :, :]
    nuclei = im == 3
    plates = im == 1
    
    nuclei_props = regionprops(label(nuclei), intensity_image=im)
    plates_props = regionprops(label(plates), intensity_image=im)
    
    
    for nucleus in np.arange(len(nuclei_props)):
        nuclei_ecce.append(nuclei_props[nucleus].eccentricity)
        nuclei_area.append(nuclei_props[nucleus].area)
        nuclei_maax.append(nuclei_props[nucleus].axis_major_length)
        nuclei_miax.append(nuclei_props[nucleus].axis_minor_length)
        nuclei_extnt.append(nuclei_props[nucleus].extent)
        nuclei_peri.append(nuclei_props[nucleus].perimeter)
        nuclei_carea.append(nuclei_props[nucleus].area_convex)
        nuclei_feret.append(nuclei_props[nucleus].feret_diameter_max)
        
    for plate in np.arange(len(plates_props)):
        plates_ecce.append(plates_props[plate].eccentricity)
        plates_area.append(plates_props[plate].area)
        plates_maax.append(plates_props[plate].axis_major_length)
        plates_miax.append(plates_props[plate].axis_minor_length)
        plates_extnt.append(plates_props[plate].extent)
        plates_peri.append(plates_props[plate].perimeter)
        plates_carea.append(plates_props[plate].area_convex)
        plates_feret.append(plates_props[plate].feret_diameter_max)
    
    print(plane)


nuclei_props = pd.DataFrame({'eccentricity':nuclei_ecce, 
                             'area':        nuclei_area,
                             'major':       nuclei_maax,
                             'minor':       nuclei_miax,
                             'extent':      nuclei_extnt,
                             'perimeter':   nuclei_peri,
                             'conv_area':   nuclei_carea,
                             'feret':       nuclei_feret})

n = nuclei_props.loc[(nuclei_props.area > pars.nucsize_min)]\
                .loc[ (nuclei_props.area < pars.nucsize_max)]

plates_props = pd.DataFrame({'eccentricity':plates_ecce, 
                             'area':        plates_area,
                             'major':       plates_maax,
                             'minor':       plates_miax,
                             'extent':      plates_extnt,
                             'perimeter':   plates_peri,
                             'conv_area':   plates_carea,
                             'feret':       plates_feret})

p = plates_props.loc[(plates_props.area > pars.nucsize_min)]\
                .loc[ (plates_props.area < pars.nucsize_max)]

n['label'] = 0
p['label'] = 1

combined = pd.concat([n,p])
Y = combined.label
combined.drop(columns=['label'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
                                                    combined, Y, 
                                                    test_size=0.33, 
                                                    random_state=42
                                                    )
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

import pickle

# save the model to disk
filename = 'meta_plate.model'
pickle.dump(clf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)

# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# kernel = 1.0 * RBF(1.0)
# gpc = GaussianProcessClassifier(kernel=kernel,
#                                 random_state=0).fit(X_train, y_train)
# print(gpc.score(X_train, y_train))
# probs = gpc.predict_proba(X_test)
