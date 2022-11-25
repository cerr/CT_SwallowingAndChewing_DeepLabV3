# Main function to run segmentation of chewing structures
# Aditi Iyer
# iyera@mskcc.org
# Jan 2, 2020

import sys
import os 
if 'MKL_NUM_THREADS' in os.environ:
   del os.environ['MKL_NUM_THREADS']

import structures.fuse_seg_mm_pm
from time import process_time

def main():
    
    # Segment chewing structures
    structures.fuse_seg_mm_pm.main(sys.argv)
    
main()
