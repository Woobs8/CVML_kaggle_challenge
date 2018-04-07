#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

def write_predictions_file(predictions, output_file):
    dir_name = os.path.dirname(output_file)
    if dir_name != "" and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    num_samples = len(predictions)
    ids = np.arange(1,num_samples+1)
    np.savetxt(output_file, np.stack((ids,predictions),axis=1), delimiter=',', header='ID,Label',fmt='%d')