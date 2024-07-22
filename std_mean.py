import gdal
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import pdb

#landsat = skimage.io.imread('/data/zlx/cross_sr/xian_landsat20200221.tif')
#landsat = landsat[:,:,[0,1,2,3,4,5,6]]
#std = []
#mean = []
#for i in range(landsat.shape[2]):
#    mean.append(np.nanmean(landsat[:,:,i]))
#    std.append(np.nanstd(landsat[:,:,i]))
#print('Landsat: std {}, mean {}'.format(std, mean))

dataframe = pd.read_csv('./sentinel_ours_all.csv')
used_imgs = dataframe.image_path.values

std = [0, 0, 0, 0, 0, 0]
mean = [0, 0, 0, 0, 0, 0]

for img in tqdm(used_imgs):
    sentinel = gdal.Open(img).ReadAsArray()
    mean += np.nanmean(sentinel, axis=(1,2))/len(used_imgs)
    std += np.nanstd(sentinel, axis=(1,2))/len(used_imgs)

print('Sentinel: std {}, mean {}'.format(std, mean))
