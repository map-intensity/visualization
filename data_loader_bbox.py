import numpy as np
import cv2
import os
import random
import pandas as pd
from scipy.ndimage import gaussian_filter
from natsort import natsorted

class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False, imagenumber = -1):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename))]

        
        self.data_files = natsorted(self.data_files)
        if (imagenumber > 0):
            self.data_files = self.data_files[:imagenumber]
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = range(0,self.num_samples)
        if self.pre_load:
            print ('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:
                
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                ht_1 = int((ht/4)*4)
                wd_1 = int((wd/4)*4)
                img = cv2.resize(img,(wd_1,ht_1))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))

                Main_Zeros = np.zeros((ht, wd), dtype=np.float32)

                Coordinates = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).to_numpy()                        
                Coordinates  = Coordinates.astype(np.float32, copy=False)
                for bbox_left, bbox_top, bbox_w, bbox_h, bbox_weight in Coordinates:
                    for i in range(int(bbox_left),int(bbox_left+bbox_w)):
                        for j in range(int(bbox_top),int(bbox_top+bbox_h)):
                            Main_Zeros[j, i] += bbox_weight
                den = gaussian_filter(Main_Zeros,sigma=6,truncate=6*6)

                if self.gt_downsample:
                    wd_1 = int(wd_1/4)
                    ht_1 = int(ht_1/4)
                    den = cv2.resize(den,(wd_1,ht_1))                
                    den = den * ((wd*ht)/(wd_1*ht_1))
                else:
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                    
                den = den.reshape((1,1,den.shape[0],den.shape[1]))
            
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                self.blob_list[idx] = blob
                idx = idx+1
                if idx % 5 == 0:                    
                    print ('Loaded ', idx, '/', self.num_samples, 'files')
               
            print ('Completed Loading ', idx, 'files')
        
        
    def __iter__(self):
        if self.shuffle:            
            if self.pre_load:            
                random.shuffle(list(self.id_list))      
            else:
                random.shuffle(list(self.data_files))
        files = self.data_files
        id_list = self.id_list
       
        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]    
                blob['idx'] = idx
            else:                    
                fname = files[idx]
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                ht_1 = (ht/4)*4
                wd_1 = (wd/4)*4
                img = cv2.resize(img,(wd_1,ht_1))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
                den  = den.astype(np.float32, copy=False)
                if self.gt_downsample:
                    wd_1 = wd_1/4
                    ht_1 = ht_1/4
                    den = cv2.resize(den,(wd_1,ht_1))                
                    den = den * ((wd*ht)/(wd_1*ht_1))
                else:
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                    
                den = den.reshape((1,1,den.shape[0],den.shape[1]))            
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
                
        
            
        
