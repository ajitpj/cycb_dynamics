#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:54:07 2023

@author: ajitj
"""

from skimage.morphology import closing, erosion
from skimage.morphology import square
from skimage.measure import label, regionprops
import numpy as np


def exportsavedTracks_to(savedTracks: dict, max_length: int):
    '''
    Exports the saved tracks dictionary as a numpy array

    Parameters
    ----------
    savedTracks : dict
        cycb_data dictionary with btrack IDs as keys and the
        (sig_arr, prediciton) tuple as the output where,
        sig_arr = cycb.measureFluorescence function as the data.
        prediction = 0 or 1 prediction for metaphase plate shape
    max_length  : int
        max_length = Number of datapoints in the image

    Returns
    -------
    data_array: numpy array with the following dimensions
                dim 0 = Track serial index
                dim 1 = Cycb measurements
                dim 3 = classification + the six columns of the 
                        cycb.measureFluorescence function

    '''
    numTracks = len(savedTracks)
    data_array     = np.zeros((numTracks, max_length, 7))
    
    for index, key in enumerate(savedTracks.keys()):
        tracklength = savedTracks[key][0].shape[0]
        data_array[index, 0:tracklength, 0]    = savedTracks[key][1]
        data_array[index, 0:tracklength, 1::]  = savedTracks[key][0]
    
    return data_array

def getROI(targetstack, tracklet, roisize):
    '''
    Uses tracklet to return roi from the targetstack.
    targetstack = image stack containing raw data
    tracklet    = tracklet from btrack output
    roisize     = Half-size for the roi 
    strel       = structuring element for creating ring around the nucleus
    '''
    
    imwidth  = targetstack.shape[1]
    imheight = targetstack.shape[2]
    tracklength = len(tracklet)
    
    y = np.array(tracklet.x, dtype=int)
    x = np.array(tracklet.y, dtype=int)
    
    roi_stack = np.zeros((tracklength, 2*roisize, 2*roisize))
    
    for t in np.arange(tracklength):
        roi_stack[t, :, :] = targetstack[t, x[t]-roisize:x[t]+roisize, \
                                            y[t]-roisize:y[t]+roisize]

    return roi_stack

def dispRects_fortrack(tracks, trackID: int, roisize: int):
    '''
    Creates squares in the selected framenumber from the centroid in the 
    selected btrack tracklet.
    INPUTS:
        tracks      = btrack ordered dictionary structure
        trackID     = btrack track ID
        roisize     = size of the square
    OUTPUTs:
        vertex_list = numpy array: v[0]=t
                                   v[1]=upper x, v[2]=upper y
                                   v[3]=lower x, v[4]=lower y
    '''
    
    tracklet = tracks[trackID]
    vertex_list = np.zeros((len(tracklet), 5))
    vertex_list[:,0] = np.array(tracklet.t[:])
    vertex_list[:,1] = np.array(tracklet.y[:], dtype=int) - roisize
    vertex_list[:,2] = np.array(tracklet.x[:], dtype=int) - roisize
    vertex_list[:,3] = np.array(tracklet.y[:], dtype=int) + roisize
    vertex_list[:,4] = np.array(tracklet.x[:], dtype=int) + roisize
    
    return vertex_list

def dispRects_all(tracks, IDs: list, roisize: int):
    '''
    Creates squares in the selected framenumber from the centroid in the 
    selected btrack tracklet.
    INPUTS:
        tracks      = btrack ordered dictionary structure
        IDs         = list of btrack track IDs
        roisize     = size of the square
    OUTPUTs:
        firstframe    = tpoint for the first frame
        vertex_list = numpy array: v[0]=upper x, v[1]=upper y
                                   v[2]=lower x, v[3]=lower y
    '''
    firstframe = []
    vertex_list = np.zeros((len(IDs),4))
    i = 0
    for id in IDs:
        tracklet = tracks[id]
        firstframe.append(tracklet.t[0])
        
        vertex_list[i,0] = int(tracklet.y[0]) - roisize
        vertex_list[i,1] = int(tracklet.x[0]) - roisize
        vertex_list[i,2] = int(tracklet.y[0]) + roisize
        vertex_list[i,3] = int(tracklet.x[0]) + roisize
        i += 1
    
    return firstframe, vertex_list

def filterby_area(mask, nucsize_min: int, nucsize_max: int):
    '''
    Filters nuclei based on their size
    Required to filter out unaligned chromosomes and prevent
    btrack from tracking them as additional objects
    mask        = segmented nuclear stain image 
    nucsize_min = Lower nuclear size threshold 
    nucsize_max = Higher nuclear size threshold
    The lower threshold is an important parameter
    '''
    # mask   = erosion(mask)
    try:
        assert(nucsize_min > 0 and nucsize_max > 0)
    except:
        raise AssertionError('Sizes must be > 0')
    
    if nucsize_min > nucsize_max:
        print('Swapping max and min size values!')
        nucsize_min, nucsize_max = nucsize_max, nucsize_min
    
    labels = label(mask)
    props  = regionprops(labels)
    labelvals = np.array([prop.label for prop in props])
    
    #Filter and store
    maskfiltered = np.zeros_like(mask, dtype=bool)
    
    for i in np.arange(len(props)):
        if props[i].area>nucsize_min and props[i].area<nucsize_max:
            maskfiltered += mask==labelvals[i]
    
    return maskfiltered


def drawAnnulus(kernel_size:int, inner_rad:int, outer_rad:int, center:tuple):
    '''kernel_size = size of the mask
    inner_rad = inner radius of the annulus
    outer_rad = outer radius of the annulus
    center = center of the annulus kernel_size//2 - 1'''
    
    if inner_rad > outer_rad:
        outer_rad, inner_rad = inner_rad, outer_rad
        
        
    mask   = np.zeros((kernel_size,kernel_size), dtype=int)

    x, y = np.ogrid[0:kernel_size,0:kernel_size]

    outer = (center[0] - x)**2 + (center[1] - y)**2 <= outer_rad**2
    inner = (center[0] - x)**2 + (center[1] - y)**2 <= inner_rad**2
    mask[outer!=inner] = 1

    strel = square(2)
    mask = closing(mask, strel)
    mask = erosion(mask, strel)
    return mask


def binImage(img, new_shape, method):
    '''
    img = Original array to be binned
    new_shape = final desired shape of the array
    method = 'min' - minimum binned 
             'max' - max. binned
             'mean' - mean binned; default
    # Copied from: https://scipython.com/blog/binning-a-2d-array-in-numpy/
    '''
    shape = (new_shape[0], img.shape[0] // new_shape[0],
             new_shape[1], img.shape[1] // new_shape[1])
    img = img.reshape(shape)
    
    if   method == 'min':
            out = img.min(-1).min(1)
    elif method == 'max':
            out = img.max(-1).max(1)
    elif method == 'mean':
            out = img.mean(-1).mean(1)
            
    return out