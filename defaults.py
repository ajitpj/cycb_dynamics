#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:55:36 2023
@author: ajitj

Default parameter values stored for convenience.
"""
from skimage.morphology import square
from utils import drawAnnulus
import pickle

class DefaultPars:
    def __init__(self):
        # Not currently binning the images. should try to reduce time.
        self.binsize = 1 
        # roughly the small diameter of a nucleus
        self.celldiameter = 71 
        # Structuring element used for morphometric functions
        self.strel_cell = square(self.celldiameter)
        # Multiply the std. of convolution before subtracting
        self.multiplier = 1.0 
        # Based on a small sample of HeLa cells, 
        # 750 < nuc_area < 2600 before thresholding
        self.nucsize_min = 350 
        self.nucsize_max = 2800
        
        
        # Default parameters for extraction and measurement
        self.roisize = 50
        self.ring_thickness = 10
        # Limit signal pixels to a circular region of 60 pix diameter
        self.cellrad = 30 
        self.circle_roi = drawAnnulus(self.roisize*2, 0, 
                                      self.cellrad, 
                                      (self.roisize,self.roisize)
                                      )
    
        # For phase image convolution
        self.strel51  = square(51)
        self.strel11  = square(11)
        self.strel7   = square(7)
        self.strel5   = square(5)
        self.strel3   = square(3)
        self.kernel   = drawAnnulus(81,32,35,(40,40))
        #drawAnnulus(41,15,20,(20,20)) for binsize = 2
        self.binsize    = 1
        self.cvl_mult   = 1.8
        self.cvl_thresh = 0.18 # > 0.2 == mitotic cell
        
        # btrack configuration
        # features to be calculated from image data
        self.btrack_config_file = 'cell_config.json'
        self.features = ["area", "major_axis_length", "minor_axis_length",
                    "orientation", "solidity",]

        self.tracking_updates = [
            "motion",
            "visual",
        ]
        
        # Shape prediction model
        filename = 'meta_plate_withint.model'#'meta_plate.model'
        self.loaded_model = pickle.load(open(filename, 'rb'))