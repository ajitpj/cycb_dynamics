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
import cycb, utils
from trackAnalysis import summarizetracks, convertTracks
from cycbGUI_v9 import Curate_tracks
from select_files import select_files

from defaults import DefaultPars
pars = DefaultPars()

# Set file names for later recall
wdir = os.getcwd()

if __name__ == "__main__":
    files = select_files()
    files.show()


#%%
path = os.path.split(files.phase_file)[0]
phase_name = os.path.split(files.phase_file)[1]
red_name   = os.path.split(files.red_file)[1]
green_name = os.path.split(files.green_file)[1]

red = imread(files.red_file)
green = imread(files.green_file)

#%%
# Read stack attributes

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

analysis_folder = path+'/%s_analysis'%phase_name[:-4]
os.mkdir(analysis_folder)
trackfile = os.path.join(analysis_folder, red_name[:-4]+'_tracks.h5')
#%%
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
    tracker.track(step_size=10)
    # generate hypotheses and run the global optimizer
    tracker.optimize()
    # store the data in an HDF5 file
    tracker.export(trackfile, obj_type='obj_type_1')
    # get the tracks as a python list
    tracks = tracker.tracks
    # optional: get the data in a format for napari
    data, properties, graph = tracker.to_napari()


###
#%%
# Shortlist tracks based on user requirements
track_summary = summarizetracks(tracks)
track_summary['duration'] = track_summary['tend'] - track_summary['tstart']
shortlist = track_summary.loc[(track_summary['duration'] > 1)]\
    .loc[(track_summary['tstart'] < 150)]

IDs = shortlist.ID.to_list()
tracks_dict = convertTracks(tracks)
# Further shortlist by discarding cells too close to the edge
edgeIDs = []
for ID in IDs:

    y = np.array(tracks_dict[ID].x, dtype=int)
    x = np.array(tracks_dict[ID].y, dtype=int)

    if (x.min() < pars.roisize+1 or x.max() > imwidth - pars.roisize - 1 or\
        y.min() < pars.roisize+1 or y.max() > imheight - pars.roisize -1):

        edgeIDs.append(ID)

nonEdgeIDs = set(IDs) - set(edgeIDs)
shortlistIDs = list(nonEdgeIDs)
shortlistIDs = np.array(shortlistIDs)
###
# Save data objects for later analysis

#%%
# Open viewer and display stacks

viewer = napari.Viewer()
viewer.add_image(green, colormap='turbo', contrast_limits=[200, 400])
viewer.add_image(red, opacity=0.5, colormap='gray', contrast_limits=[80, 400])
viewer.add_tracks(data, properties=properties, graph=graph)

# Add the GUI to curate tracks
# del Curate_tracks
# from cycbGUI_v9 import Curate_tracks
if __name__ == "__main__":
    v = Curate_tracks(green, redseg, red, shortlistIDs, tracks_dict)
    viewer.window.add_dock_widget(v)
    v.show()


#%%
# Calculate the floor of the green channel by looking at empty pixels
# in the phase image

phase = imread(files.phase_file)
green_bkg_data  = np.zeros((planes, 3))
print('Calculating the green channel floor...')
for plane in np.arange(planes):
    # calculate the green image floor
    bkg_mask = cycb.calculateBKG(phase, plane)
    greenimg = green[plane, :, :]
    nonzeros = greenimg[np.where(bkg_mask == 1)]
    green_bkg_data[plane, 0] = nonzeros.mean()
    green_bkg_data[plane, 1] = nonzeros.std()
    green_bkg_data[plane, 2] = len(nonzeros)
    
    print('Finished plane# %d' % plane)

#%% Save the data
raw_data = v.cycb_data
cycb_data = utils.exportsavedTracks_to(raw_data, 170)

to_save = {'cycb_dat':       cycb_data,
           'green_bkg':      green_bkg_data,
           'defaults':       pars,
           'track_summary':  track_summary,
           'shortlisted':    shortlist,
           'saved_trackIDs': nonEdgeIDs
           }  
save_file = os.path.join(analysis_folder, phase_name[:-4]+'.analysis')
import pickle
with open(save_file, 'wb') as file:
    pickle.dump(to_save, file)

