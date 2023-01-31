#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:17:05 2023

@author: ajitj
"""
from magicclass import magicclass, MagicTemplate
from magicclass.widgets import PushButton
from magicclass import field

@magicclass
class select_files(MagicTemplate):
    
    @magicclass(layout="vertical")
    class Frame1:
        phase_file = field(str, widget_type="FileEdit", 
                           options={"filter": '*.tif', "mode": 'r',
                                    "label": 'Select phase file'})
        red_file   = field(str, widget_type="FileEdit", 
                           options={"filter": '*.tif', "mode": 'r',
                                    "label": 'Select red file'})
        green_file = field(str, widget_type="FileEdit", 
                           options={"filter": '*.tif', "mode": 'r',
                                    "label": 'Select green file'})
            
    @magicclass(layout="horizontal")
    class Frame2:
        but1 = field(PushButton, name='Finalize selections')
    
    
    
    def __init__(self):
         super().__init__()
         self.red_file     = ''
         self.green_file   = ''
         self.phase_file   = ''
    
    @Frame2.but1.connect
    def _okselection(self):
        self.red_file     = self.Frame1.red_file.value
        self.green_file   = self.Frame1.green_file.value
        self.phase_file   = self.Frame1.phase_file.value
        select_files.close(self)
