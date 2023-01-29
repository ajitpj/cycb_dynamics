from magicclass import magicclass, MagicTemplate
from magicclass.widgets import Figure, PushButton, Label
from magicclass import field
from napari.types import ImageData, ArrayLike
import cycb
from defaults import DefaultPars
import utils
import numpy as np
import pandas as pd

from skimage.filters import threshold_otsu
from skimage.morphology import white_tophat, erosion, square
    
@magicclass
class Curate_tracks(MagicTemplate):
    
    @magicclass(layout="horizontal")
    class Frame1:
        TrackNum = field(str, widget_type="SpinBox",
                         options={"min": 0, "max": 2, "step": 1})
            
    @magicclass(layout="horizontal")
    class Frame2:
        but1 = field(PushButton)
        but2 = field(PushButton)
        but3 = field(PushButton)
    
    plt = field(Figure, options={"nrows": 1, "ncols": 1})
   
    def __init__(self, target: ImageData, 
                       mask: ImageData,
                       red: ImageData,
                       IDs: ArrayLike, 
                       tracks: ArrayLike):
         super().__init__()
         
         self.pars    = DefaultPars()
         self.IDs     = IDs    # Track IDs assigned in btrack
         self.target  = target # Target stack to be measured
         self.mask    = mask   # Mask stack
         self.red   = red  # Phase image for convolution 
         self.tracks  = tracks   # Track data for TrackID from btrack
         self.numtrks = len(IDs)
         self.imshape = target.shape
         
         self.cycb_data = dict()
 
    def __post_init__(self, ):
         self.Frame2.but1.text = 'Dislpay'
         self.Frame2.but2.text = 'Save'
         self.Frame2.but3.text = 'Next'
         self.plt.min_height   = 400
         self.Frame1.TrackNum.max = self.numtrks - 1
         
    
    @Frame1.TrackNum.connect
    @Frame2.but1.connect
    def _displayTrack(self):
        
        selectedTrack = int(self.Frame1.TrackNum.value)
        
        currentID    = self.IDs[selectedTrack]
        currentTrack = self.tracks[currentID]
        
        # Calculate the CycB signal
        self.sig_arr, _, _ = cycb.measureCell(self.target, self.mask, 
                                          currentTrack, 
                                          self.pars.roisize, 
                                          self.pars.ring_thickness, 
                                          self.pars.cellrad)
        
        # CLassify shape; metaphase plate = 1
        mask_roi = utils.getROI(self.red, currentTrack, self.pars.roisize)
        mask_roi = cycb.segmentROI(mask_roi, self.pars.strel7)
        props = cycb.getregProps(mask_roi[0,:,:], 'largest')
        for i in np.arange(1, mask_roi.shape[0]):
            props = pd.concat([props, 
                               cycb.getregProps(mask_roi[i,:,:], 'largest')
                              ]
                              )
        
        self.prediction = self.pars.loaded_model.predict(props)
        
        # Display update
        new_cam_center = (0, currentTrack.y[0], currentTrack.x[0])
        self.parent_viewer.camera.center = new_cam_center
        self.parent_viewer.camera.zoom   = 3
        self.parent_viewer.cursor.position = new_cam_center
        
        
        if self.parent_viewer.layers[-1].name == 'current cell':
            shapes_layer = self.parent_viewer.layers['current cell']
        else:
            shapes_layer = self.parent_viewer.add_shapes(name = 'current cell')
        
        shapes_layer.add_rectangles(
                                    [[currentTrack.y[0] - self.pars.roisize, 
                                      currentTrack.x[0] - self.pars.roisize],
                                     [currentTrack.y[0] + self.pars.roisize, 
                                      currentTrack.x[0] + self.pars.roisize]],
                                      edge_color='red',
                                      face_color=''
                                    )
        
        self.plt.clf()
        self.plt.plot(self.sig_arr[:,0], 'r.-')
        self.plt.plot(self.sig_arr[:,3], 'b.-')
        meta = np.nonzero(self.prediction)
        self.plt.plot(meta, self.sig_arr[meta, 3], 'go')
        # self.plt.plot(mitosisflag)
        self.plt.legend(['Cytoplasmic', 'Nuclear'])
        self.plt.xlabel('time (x5 min)')
        self.plt.ylabel('mNG-CycB a.u.')
        self.plt.grid('on')
        
        return
    
    
    @Frame2.but2.connect
    def _saveTrack(self):
        selectedTrack = int(self.Frame1.TrackNum.value)
        currentID    = self.IDs[self.IDs[selectedTrack]]
        self.cycb_data[currentID] = (self.sig_arr, self.prediction)
        self.Frame1.TrackNum.value  =  str(selectedTrack+1)
        
    
    @Frame2.but3.connect
    def _nextTrack(self):
        selectedTrack = int(self.Frame1.TrackNum.value)
        self.Frame1.TrackNum.value  =  str(selectedTrack+1)
        
           
