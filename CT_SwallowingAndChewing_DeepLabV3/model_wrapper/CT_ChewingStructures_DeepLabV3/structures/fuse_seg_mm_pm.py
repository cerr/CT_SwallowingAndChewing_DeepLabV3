# Author: Aditi Iyer
# Email: iyera@mskcc.org
# Date: Oct 25, 2019
# Description: Script to generate segmentations of masseters (left and right) and medial pterygoids (left and right)
#             by fusing probability maps  from three different views.
# Architecture :  Deeplab v3+ with resnet backbone
# Usage: python fuse_seg_mm_pm.py [argv]
# Output: 2D masks saved as h5 files to output folder

from time import process_time

import h5py
import numpy as np
import os

from . import run_seg_mm_pm_ax
from . import run_seg_mm_pm_cor
from . import run_seg_mm_pm_sag


def main(argv):
    
    # Define paths
    inputH5Path = argv[1]
    outputH5Path = argv[2]
    batchSize = int(argv[3])

    inputH5PathAx = os.path.join(inputH5Path, 'axial')
    inputH5PathSag = os.path.join(inputH5Path, 'sagittal')
    inputH5PathCor = os.path.join(inputH5Path, 'coronal')

    # Get probability maps from different views
    print('Beginning inference...')
    t0 = process_time()
    probMapAx, fName = run_seg_mm_pm_ax.main(inputH5PathAx, batchSize)
    print(process_time() - t0)
   
    t1 = process_time()
    probMapSag = run_seg_mm_pm_sag.main(inputH5PathSag, batchSize)
    print(process_time() - t1)
   
    t2 = process_time()
    probMapCor = run_seg_mm_pm_cor.main(inputH5PathCor, batchSize)
    print(process_time() - t2)

    # Fuse maps
    print('Generating consensus segmentations...')
    t3 = process_time()
    avgProb = (probMapAx + probMapSag + probMapCor) / 3
    labels = np.argmax(avgProb, axis=0)
    print(process_time() - t3)    

    # Write to HDF5
    print('Writing output h5 files to disk...')
    t4 = process_time()
    numSlc = labels.shape[2]
    path, file = os.path.split(fName)
    idx = file.find('_')
    prefix = file[0:idx]
    for iSlc in range(numSlc):
        maskfilename = prefix + '_slice_' + str(iSlc + 1) + '.h5'
        mask = labels[:, :, iSlc]
        with h5py.File(os.path.join(outputH5Path, maskfilename), 'w') as hf:
            hf.create_dataset("mask", data=mask)
    print(process_time()-t4)

    
    return labels
