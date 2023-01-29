#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:08:37 2023

@author: ajitj
"""

import os
from skimage.io import imread
import napari
import btrack
import numpy as np
import pandas as pd
import cycb, utils
from trackAnalysis import summarizetracks
from cycbGUI_v9 import Curate_tracks

from defaults import DefaultPars
pars = DefaultPars()

# Set file names for later recall
wdir = os.getcwd()
phase_file = '../20221006_D12_1.tif'
green_file = '../20221006_corrected_green_D12_1.tif'
red_file = '../20221006_red_uncalibrated_D12_1.tif'

# Read stack attributes
red = imread(red_file)
green = imread(green_file)
# phase = imread(phase_file)
if not green.shape == red.shape:
    raise ValueError('RFP and GFP stack size mismatch, can\'t proceed!')

imwidth  = red.shape[1]
imheight = red.shape[2]
planes   = red.shape[0]

redseg   = np.zeros_like(red)  # Used to retrieve ROIs from green
trackseg = np.zeros_like(red)  # Used for tracking

print('Now segmenting DNA images...')
for plane in np.arange(planes):
    # DNA stain thresholding
    im = red[plane, :, :].copy()
    redseg[plane, :, :], trackseg[plane, :, :] \
        = cycb.preprocess(red, plane, pars.strel_cell)
    print('Finished plane# %d' % plane)


# btrack copied code

objects = btrack.utils.segmentation_to_objects(
    trackseg, properties=tuple(pars.features))

with btrack.BayesianTracker() as tracker:

    # configure the tracker using a config file
    tracker.configure(pars.btrack_config_file)
    tracker.features = pars.features
    # append the objects to be tracked
    tracker.append(objects)
    # tell the tracker to use certain information while
    # performing tracking
    tracker.track(tracking_updates=pars.tracking_updates)
    # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
    tracker.volume = ((0, 2048), (0, 2048))
    # track them (in interactive mode)
    tracker.track(step_size=100)
    # generate hypotheses and run the global optimizer
    tracker.optimize()
    # store the data in an HDF5 file
    tracker.export(red_file+'_tracks.h5', obj_type='obj_type_1')
    # get the tracks as a python list
    tracks = tracker.tracks
    # optional: get the data in a format for napari
    data, properties, graph = tracker.to_napari()


###
# Shortlist tracks based on user requirements
track_summary = summarizetracks(tracks)
track_summary['duration'] = track_summary['tend'] - track_summary['tstart']
shortlist = track_summary.loc[(track_summary['duration'] > 20)]\
    .loc[(track_summary['tstart'] < 50)]

IDs = shortlist.ID.to_numpy()

# Further shortlist by discarding cells too close to the edge
nonEdgeIDs = []
for ID in np.arange(len(shortlist)):

    y = np.array(tracks[int(shortlist.iloc[ID].ID)].x, dtype=int)
    x = np.array(tracks[int(shortlist.iloc[ID].ID)].y, dtype=int)

    if (x.min() > pars.roisize) and (x.max() < imwidth - pars.roisize) and\
       (y.min() > pars.roisize) and (y.max() < imheight - pars.roisize):

        nonEdgeIDs.append(ID)

nonEdgeIDs = np.array(nonEdgeIDs)

###
# Save data objects for later analysis


# Open viewer and display stacks
redseg = redseg.astype('uint')
viewer = napari.Viewer()
viewer.add_image(green, colormap='turbo', contrast_limits=[200, 400])
viewer.add_image(red, opacity=0.5, colormap='gray', contrast_limits=[80, 400])
viewer.add_tracks(data, properties=properties, graph=graph)

# Add the GUI to curate tracks
del Curate_tracks
from cycbGUI_v9 import Curate_tracks
if __name__ == "__main__":
    a = Curate_tracks(green, redseg, red, nonEdgeIDs, tracks)
    viewer.window.add_dock_widget(a)
    a.show()

    cycb_data = utils.exportsavedTracks_to(a.cycb_data, planes)