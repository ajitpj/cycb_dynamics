#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:38:21 2023

@author: ajitj
"""
import pandas as pd
import numpy as np


def summarizetracks(tracks):
    '''
    Returns the summary dataframe for all the tracks
    INPUTS:
        tracks = tracks list output of btrack
    OUPUT:
        summary_table pandas dataframe with:
        ID = track ID from btrack
        tstart = t = 0
        tend   = t = end
        x_start = starting X
        y_start = starting Y
        gener   = generation
        parent  = parent
        root    = root
    '''
    ID      = []
    tstart  = []
    tend    = []
    xstart  = []
    ystart  = []
    parent  = []
    root    = []
    gener   = []
    for tracklet in tracks:
        ID.append(tracklet.ID)
        tstart.append(tracklet.t[0])
        tend.append(tracklet.t[-1])
        xstart.append(tracklet.x[0])
        ystart.append(tracklet.y[0])
        gener.append(tracklet.generation)
        root.append(tracklet.root)
        parent.append(tracklet.parent)
    
    summary_table = pd.DataFrame({'ID': ID,
                                  'tstart': tstart,
                                  'tend': tend,
                                  'x_start': xstart,
                                  'y_start': ystart,
                                  'gener': gener,
                                  'root': root,
                                  'parent': parent})
    return summary_table

def convertTracks(tracks):
    '''
    Function converts the btrack output 'tracks', which is a list of ordered
    dictionaries, into a dictionary with tracklet.ID as the key and the entire
    tracklet (includig the ID).

    Parameters
    ----------
    tracks : btrack output
        Output from the btrack program.

    Returns
    -------
    convertedTracks : dictionary
        a dictionary with tracklet.ID as the key and the entire
        tracklet (includig the ID)..

    '''
    
    convertedTracks = {}
    
    for i in np.arange(len(tracks)):
        convertedTracks[tracks[i].ID] = tracks[i]
    
    
    return convertedTracks
# def displayTracklet(tracklet, viewer):
#     '''
#     Displays a selected tracklet as a graph and napari layer
#     INPUTS:
#         tracklet: btrack tracklet
#         viewer  : napari viewer object
#     '''
    
#     return