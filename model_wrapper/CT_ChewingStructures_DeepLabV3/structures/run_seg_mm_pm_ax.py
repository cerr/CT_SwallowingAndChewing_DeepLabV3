# Author: Aditi Iyer
# Email: iyera@mskcc.org
# Date: Oct 10, 2019
# Description: Script to segment masseters (left and right) and medial pterygoids (left and right) from axial CT images
# Architecture :  Deeplab v3+ with resnet backbone
# Usage: python run_seg_mm_pm_ax.py [argv] [path to input 2D H5 axial files]
# Output: 3D probability map

import sys
from time import process_time

import numpy as np
import os
import torch
from dataloaders.custom_dataset import *
from modeling.deeplab import *
from skimage.transform import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

input_size = 320

def main(inputH5Path, batchSize):
    trainer = Trainer(inputH5Path,batchSize)
    probMapAll, filename = trainer.validation(inputH5Path)

    return probMapAll, filename


class Trainer(object):
    def __init__(self, inputDir, batchSize):
       
        self.inputDir = inputDir
        self.nClass = 5
        self.mean=(0.2306, 0.2306, 0.2306)
        self.std=(0.2093, 0.2093, 0.2093)       
        self.cropSize = 321
        cDir = os.path.dirname(os.path.realpath(__file__))
        pDir = os.path.dirname(os.path.dirname(os.path.dirname(cDir)))
        #modelPath = os.path.join(pDir,'model','MM_PM_Ax_model.pth.tar')
        self.modelPath = '/software/model/MM_PM_Ax_model.pth.tar'
        self.batchSize = batchSize

        #Define dataloadr
        kwargs = {'num_workers': 0, 'pin_memory': True}   
        testSet = struct(self)
        self.testLoader = DataLoader(testSet, batch_size=self.batchSize, shuffle=False, drop_last=False, **kwargs)
 
        # Define network
        print('Loading network...')
        t0 = process_time()
        self.model = DeepLab(num_classes=self.nClass,
                             backbone='resnet',
                             output_stride=16,
                             sync_bn=False,
                             freeze_bn=False)
        print(process_time() - t0)

        #Using CUDA
        print('Loading model weights...')
        t1 = process_time()
        if torch.cuda.device_count() and torch.cuda.is_available():
           print('Using GPU')
           self.cuda = True
           print('GPU device count: ',torch.cuda.device_count())
           device = torch.device("cuda:0")
           checkpoint = torch.load(self.modelPath)
           self.model.load_state_dict(checkpoint['state_dict'])
           self.model = self.model.to(device)
        else:
           self.cuda = False
           print('Using CPU')
           device = torch.device('cpu')
           checkpoint = torch.load(self.modelPath, map_location=device)
           self.model.load_state_dict(checkpoint['state_dict'])            
      
        print(process_time()-t1)
        print('Loaded.')
   
    def validation(self, inputH5Path):
       
        print('Computing probability maps...')

        self.model.eval()
       
        tbar = tqdm(self.testLoader,desc='\r')
        numImgs =  len(self.testLoader.dataset)
           
        fileList = [] 

        for i, sample in enumerate(tbar):
            
            image = sample['image']
            imgSiz = sample['imagesize']
            height = imgSiz[0][0]
            width = imgSiz[1][0]
            fileList.extend(sample['fname'])            

            if self.cuda:
              image = image.cuda()
            with torch.no_grad():
              output = self.model(image)
            
            # Get probability maps
            sm = torch.nn.Softmax(dim=1)
            prob = sm(output)

            # Reshape probability map
            probMap = np.squeeze(prob.cpu().numpy())

            pSize = np.shape(probMap)            
            if len(pSize)==3:
               probMap = np.expand_dims(probMap, 0)
               pSize = np.shape(probMap)

            if i==0:
              probMapAll = np.zeros((self.nClass,height,width,numImgs))

            for c in range(0, self.nClass):
                for slc in range(0, pSize[0]):                  
                  classProbMap = probMap[slc,c, :, :]
                  resizProbMap = resize(classProbMap,(height, width), anti_aliasing=True)
                  iNum = i*self.batchSize+slc 
                  fileName = fileList[iNum]
                  idx = fileName.find('slice')
                  slcNum = int(fileName[idx+6:-3])
                  probMapAll[c, :, :, slcNum-1] = resizProbMap
             

        return probMapAll, fileList[-1]


if __name__ == "__main__":
    main(sys.argv)
