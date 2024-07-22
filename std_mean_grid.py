from osgeo import gdal
#import gdal
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import pdb
from multiprocessing import Pool
#landsat = skimage.io.imread('/data/zlx/cross_sr/xian_landsat20200221.tif')
#landsat = landsat[:,:,[0,1,2,3,4,5,6]]
#std = []
#mean = []
#for i in range(landsat.shape[2]):
#    mean.append(np.nanmean(landsat[:,:,i]))
#    std.append(np.nanstd(landsat[:,:,i]))
#print('Landsat: std {}, mean {}'.format(std, mean))

dataframe = pd.read_csv('./grid_second_all.csv')
used_imgs = dataframe.image_path.values
count = len(used_imgs)

a=open('nan_grid_second_bands.txt', 'a+')

isnan = False

def isnone(img):
    sentinel = gdal.Open(img)
    flag = True
    if sentinel is None:
        flag = False
        a.writelines(img+'\n')
    else:
        sentinel = sentinel.ReadAsArray()
        for i in range(sentinel.shape[0]):
            if np.isnan(sentinel[i,:,:]).all():
                flag = False
                a.writelines(img+'\n')
    return flag, img

pool =Pool(processes=50)
result = pool.map(isnone, used_imgs)

a.close()
