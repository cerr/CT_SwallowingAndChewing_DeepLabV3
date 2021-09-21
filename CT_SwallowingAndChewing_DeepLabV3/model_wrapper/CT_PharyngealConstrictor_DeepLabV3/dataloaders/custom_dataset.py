# Author: Aditi Iyer
# Email: iyera@mskcc.org
# Date: Feb 10, 2020
# Description: Custom dataloader for Larynx segmentation model

import os
import numpy as np
import scipy.misc as m
from PIL import Image
import h5py
from torch.utils import data
from torchvision import transforms
from skimage.transform import resize
from dataloaders import custom_transforms as tr

class struct(data.Dataset):

    def __init__(self, args):

        self.root = args.inputDir

        self.args = args
        self.files = {}

        self.files = self.glob(rootdir=self.root, suffix='.h5')
        print("Found %d images" % (len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        _img, _imagesize, _fname = self._get_processed_img(index)
        sample = {'image': _img, 'imagesize': _imagesize, 'fname': _fname}
    
        #Mean & std dev. normalization
        return self.transform_ts(sample)

    def _get_processed_img(self, index):
        """ Load scan, resize, normalize to match pre-trained dataset """
        img_path = self.files[index].rstrip()
        _dirname, _fname = os.path.split(img_path)
        _img, _imagesize = self._load_image(img_path)

        return _img, _imagesize, _fname

    def _load_image(self, img_path):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Read H5 image
        hf = h5py.File(img_path, 'r')
        datasetName = list(hf.keys())
        datasetName = datasetName[0]
        dset = '/' + datasetName
        if datasetName=='OctaveExportScan':
            im = hf[dset]
            im = im['value']
            im = im[:]
        else:
           im = hf[dset][:]

        #Resize image
        image = np.array(im)
        image = image.reshape(im.shape).transpose()
        imagesize = np.shape(image)
        inputSize = 320
        image = resize(image, (inputSize,inputSize), anti_aliasing = True)

        #Normalize image from 0-255 (to match pre-trained dataset of RGB images with intensity range 0-255)
        image = (255*(image - np.min(image)) / np.ptp(image).astype(int)).astype(np.uint8)
        image = Image.fromarray(image.astype(np.uint8))

        return image, imagesize

    def glob(self, rootdir='.', suffix=''):
        """Performs glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(rootdir, filename)
            for filename in os.listdir(rootdir) if filename.endswith(suffix)]

    def transform_ts(self, sample):
        
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.cropSize),
            tr.Normalize(mean=self.args.mean, std=self.args.std),
            tr.ToTensor()])

        return composed_transforms(sample)

