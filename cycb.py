from skimage.morphology import white_tophat
from skimage.morphology import square, dilation
from skimage.filters import threshold_isodata, gaussian, threshold_multiotsu
from skimage.feature import match_template
from skimage.measure import label, regionprops
import numpy as np
import utils

# Default values for key parameters/variable
from defaults import DefaultPars

pars = DefaultPars()

def predictShape(im, model):
    prediction = []
    return prediction

def shapeMetric(phImg, kernel, binsize):
    '''
    Calculates the convolution of the ROI with a circular annulus.
    Returns
    -------
    shapescore : TYPE
        DESCRIPTION.

    '''
    nPlanes = phImg.shape[0]
    shapescore = np.zeros(nPlanes)
    phase_center = np.zeros((nPlanes, 2))
    
    for plane in np.arange(nPlanes):
        
        if binsize > 1:
            ph = utils.binImage(phImg, (phImg.shape[0]//binsize,
                               phImg.shape[1]//binsize), 'max')
        curreplane = phImg[plane,:,:]
        ph =  curreplane - curreplane.mean() - 2*curreplane.std()
        ph[ph<0] = 0
        ph = white_tophat(ph, pars.strel5)
        ph = dilation(ph, pars.strel3)
    
        # Cross-correlation
        cvl = match_template(ph, kernel, "same")
        cvl[cvl<0] = 0
        cvl = cvl - cvl.mean() - pars.cvl_mult * cvl.std()
        cvl[cvl<0] = 0
        
        shapescore[plane] = cvl.max()
        
        # Find centroid; will crash if no centroid?
        thresh_ph = threshold_multiotsu(cvl, 3)
        mask = cvl > thresh_ph[-1]
        props = regionprops(label(mask))
        if props[0].centroid[0] > 35 and props[0].centroid[1] > 35:
            phase_center[plane, :] = props[0].centroid
        else:
            phase_center[plane, :] = [50,50]
    
    return phase_center, shapescore


def preprocess(targetstack, plane, strel_cell):
    '''
    Preprocesses the specified plane from the targetstack
    INPUTS:
        targetstack = [t, x, y] image stack
        plane       = uint(plane number)
        strel_cell  = structured element for white_tophat
    '''
    im = targetstack[plane,:,:].copy()
    im = gaussian(im) # 2D gaussian smoothing filter to reduce noise
    im = white_tophat(im, strel_cell) # Background subtraction + uneven illumination correction
    thresh_im = threshold_isodata(im) # find threshold value
    redseg = im > pars.multiplier*thresh_im # only keep pixels above the threshold
    lblred = label(redseg)
    
    # remove small and large objects - unaligned chromosomes
    trackseg = utils.filterby_area(lblred, pars.nucsize_min, pars.nucsize_max)
    
    return redseg, trackseg



def measureCell(targetstack, mask, tracklet, roisize, strelsize, cellsize):
    '''
    Wrapper for measuring CycB signal in individual cells
    INPUTS:
    targetstack - Raw CycB(green) stack
    mask        - Nuclear stain segmentation
    roisize     - Size of the ROI to use
    strelsize   - Size of the structuring element
    cellsize    - Size of the circular region to limit signal measurement
    OUTPUTS:
    measurements - Array with 6 columns
    
    '''
    strel = square(strelsize)
    greenROI    = utils.getROI(targetstack, tracklet, roisize)
    maskROI     = utils.getROI(mask, tracklet, roisize)
    measurements = measureFluorescence(greenROI, maskROI, 
                                           strel, cellsize)
    
    return measurements, greenROI, maskROI


def measureFluorescence(target, mask, strel, roisize):
    '''
    Return fluorescence signal from the roi using the structured
    element to extend the mask. 
    The large ROI often contains adjacent nuclei. Therefore, uses
    the roisize to draw a circular region of interest to restrict 
    measurements to pixels within the roi.
    INPUT:
    target = [t,x,y] stack
    mask   = Segmentation of nucleus
    strel  = structured element
    method = tobeadded
    OUTPUT:
    signal = np array with statistical summary
    # column 1 = cytoplasmic average
    # column 2 = cytoplasmic std. dev.
    # column 3 = cytoplasmic Number of pixels
    # column 4 = chromatin average
    # column 5 = chromatin std. dev.
    # column 6 = chromatin Number of pixels
    '''

    
    if len(target)>0:
        tpoints = target.shape[0]
        imwidth = target.shape[1]
        
        center = np.floor(imwidth/2)
        circle_roi = utils.drawAnnulus(imwidth, 0, roisize, (center,center))
        
        signal  = np.zeros((tpoints, 6))
        
        for t in np.arange(tpoints):
            # Cytoplasmic pixels
            mask_t = mask[t,:,:].copy()
            mask_t = mask_t * circle_roi
            cyto_mask = dilation(mask_t, strel) - mask_t
            cyto_reg  = cyto_mask * target[t, :, :]
            cyt_pix   = cyto_reg[cyto_reg>0]
            signal[t,0] = cyt_pix.mean()
            signal[t,1] = cyt_pix.std()
            signal[t,2] = len(cyt_pix)
            # Chromatin pixels
            nuc_reg  = mask_t * target[t, :, :]
            nuc_pix  = nuc_reg[nuc_reg>0]
            signal[t,3] = nuc_pix.mean()
            signal[t,4] = nuc_pix.std()
            signal[t,5] = len(nuc_pix)
    else:
        signal = np.zeros((1,6))
    
    return signal
    


