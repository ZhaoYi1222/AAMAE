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

dataframe = pd.read_csv('./grid_second_all.csv')
used_imgs = dataframe.image_path.values
count = len(used_imgs)

std = [0, 0, 0, 0, 0, 0, 0]
mean = [0, 0, 0, 0, 0, 0, 0]

a=open('nan_grid_second_bands.txt', 'w')

isnan = False

for img in tqdm(used_imgs):
    if 'landsat' in img and os.path.exists(img):
        sentinel = gdal.Open(img).ReadAsArray()
        for i in range(sentinel.shape[0]):
            if np.isnan(sentinel[i,:,:]).all():
                a.writelines('band {} of img {} is all nan\n'.format(i, img))
                isnan = True
        if isnan:
            count-=1
            isnan = False
            continue
        mean += np.nanmean(sentinel, axis=(1,2))#/len(used_imgs)
        std += np.nanstd(sentinel, axis=(1,2))#/len(used_imgs)
    else:
        count-=1
        isnan = False

print(count)
a.close()

print('Landsat: std {}, mean {}'.format(std/count, mean/count))
