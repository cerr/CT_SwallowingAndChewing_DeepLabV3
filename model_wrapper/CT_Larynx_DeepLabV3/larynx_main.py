# Main function to run segmentation of larynx
# Aditi Iyer
# iyera@mskcc.org
# Jan 2, 2020

import sys

import os
if 'MKL_NUM_THREADS' in os.environ:
   del os.environ['MKL_NUM_THREADS']

import structures.fuse_seg_larynx

def main():
    # Segment larynx
    structures.fuse_seg_larynx.main(sys.argv)


main()
