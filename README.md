# CT_SwallowingAndChewing_DeepLabV3

This is a conda archive containing DeepLabV3+ models<sup>[1]</sup> and dependencies for deployment on Linux OS in conjunction with [CERR](https://github.com/cerr/CERR/)<sup>[2][3]</sup>.    

Auto-segmented structures include 
* Masseters (left and right)
* Medial pterygoids (left and right)
* Larynx 
* Pharyngeal constrictor muscle. 

A [Jupyter notebook](https://github.com/cerr/CT_SwallowingAndChewing_DeepLabV3/blob/master/demo_DLseg_swallowing_and_chewing_structures.ipynb) is provided to demonstrate usage.


# License
This codebase uses the GNU-GPL copyleft license (https://www.gnu.org/licenses/lgpl3.0.en.html) to allow open-source distribution with additional restrictions. The
license retains the ability to propagate any changes to the codebase back to the opensource community along with the following restrictions (i) No Clinical Use, (ii) No
Commercial Use, and (iii) Dual Licensing which reserve the right to diverge and/or modify and/or expand the model implementations library to have a closed
source/proprietary version along with the open source version in future. It should be noted that the segmentation models are distributed strictly for research use. Clinical
or commercial use is prohibited. CERR and containerized model implementations have not been approved by the U.S. Food and Drug Administration (FDA). The library merely
provides implementations of the developed models, whereas the creators of models retain the copyright to their work.


# Citation
[1] Iyer A, Thor M, Haq R, Deasy JO, Apte AP. Deep learning-based auto-segmentation of swallowing and chewing structures. bioRxiv 2019; https://www.biorxiv.org/content/10.1101/772178v1.full  
[2] Iyer, A., Locastro, E., Apte, A., Veeraraghavan, H. and Deasy, J.O., 2021. Portable framework to deploy deep learning segmentation models for medical images. bioRxiv.https://www.biorxiv.org/content/10.1101/2021.03.17.435903v1.full   
[3] Apte AP, Iyer A, Thor M, Pandya R, Haq R, Jiang J, LoCastro E, Shukla-Dave A, Sasankan N, Xiao Y, Hu YC. Library of deep-learning image segmentation and outcomes model-implementations. Physica Medica. 2020 May 1;73:190-6.  
