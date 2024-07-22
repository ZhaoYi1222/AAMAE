import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob
from typing import Any, Optional, List
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio import logging
import random
import lmdb 
import pickle
import copy
log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class CustomDatasetFromImages(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class FMoWTemporalStacked(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
    
    def __init__(self, csv_path: str, transform: Any):
        """
        Creates Dataset for temporal RGB image classification. Stacks images along temporal dim.
        Usually used for fMoW-RGB-temporal dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion
        """
        super().__init__(in_c=9)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

        self.min_year = 2002

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        splt = single_image_name_1.rsplit('/', 1)
        base_path = splt[0]
        fname = splt[1]
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit('_', 1)
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
        temporal_files = glob(regexp)
        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_tensor_1 = self.transforms(img_as_img_1)  # (3, h, w)

        img_as_img_2 = Image.open(single_image_name_2)
        img_as_tensor_2 = self.transforms(img_as_img_2)  # (3, h, w)

        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_3 = self.transforms(img_as_img_3)  # (3, h, w)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        img = torch.cat((img_as_tensor_1, img_as_tensor_2, img_as_tensor_3), dim=0)  # (9, h, w)
        return (img, single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporal(SatelliteDataset):
    def __init__(self, csv_path: str):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info.iloc[:, 2])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # hard-coded for fMoW

        mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        self.normalization = transforms.Normalize(mean, std)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]


        suffix = single_image_name_1[-15:]
        prefix = single_image_name_1[:-15].rsplit('_', 1)
        regexp = '{}_*{}'.format(prefix[0], suffix)
        regexp = os.path.join(self.dataset_root_path, regexp)
        single_image_name_1 = os.path.join(self.dataset_root_path, single_image_name_1)
        temporal_files = glob(regexp)
        
        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_img_2 = Image.open(single_image_name_2)
        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        del img_as_img_3
        img_as_tensor_1 = self.scale(img_as_tensor_1)
        img_as_tensor_2 = self.scale(img_as_tensor_2)
        img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and \
                    img_as_tensor_2.shape[2] > 224 and \
                    img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w],
                    img_as_tensor_2[..., :min_w],
                    img_as_tensor_3[..., :min_w]
                ], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and \
                    img_as_tensor_2.shape[1] > 224 and \
                    img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w, :],
                    img_as_tensor_2[..., :min_w, :],
                    img_as_tensor_3[..., :min_w, :]
                ], dim=-3)
            else:
                img_as_img_1 = Image.open(single_image_name_1)
                img_as_tensor_1 = self.totensor(img_as_img_1)
                img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_1, img_as_tensor_1], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor
        img_as_tensor_1 = self.normalization(img_as_tensor_1)
        img_as_tensor_2 = self.normalization(img_as_tensor_2)
        img_as_tensor_3 = self.normalization(img_as_tensor_3)

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len



class CustomDatasetFromOursTemporal(SatelliteDataset):
    def __init__(self, csv_path: str):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info['image_path'])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info['category'])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info['timestamp'])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # hard-coded for fMoW

        mean = [0.075134, 0.08517814, 0.07818352]
        std = [0.02362014, 0.02170983, 0.02141723]
        
        self.normalization = transforms.Normalize(mean, std)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]
        
        if 's2ng' in single_image_name_1:
            matched_img = single_image_name_1.replace('s2ng', 's2g')
            if matched_img in self.image_arr:
                temporal_files = [matched_img]
            else:
                temporal_files = []
        else:
            matched_img = single_image_name_1.replace('s2g', 's2ng')
            if matched_img in self.image_arr:
                temporal_files = [matched_img]
            else:
                temporal_files = []

        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]

        img_as_img_1 = self.open_image(single_image_name_1)[:, :, [2,1,0]]
        img_as_img_2 = self.open_image(single_image_name_2)[:, :, [2,1,0]]
        img_as_img_3 = self.open_image(single_image_name_3)[:, :, [2,1,0]]
        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        del img_as_img_3
        img_as_tensor_1 = self.scale(img_as_tensor_1)
        img_as_tensor_2 = self.scale(img_as_tensor_2)
        img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and \
                    img_as_tensor_2.shape[2] > 224 and \
                    img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w],
                    img_as_tensor_2[..., :min_w],
                    img_as_tensor_3[..., :min_w]
                ], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and \
                    img_as_tensor_2.shape[1] > 224 and \
                    img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w, :],
                    img_as_tensor_2[..., :min_w, :],
                    img_as_tensor_3[..., :min_w, :]
                ], dim=-3)
            else:
                img_as_img_1 = self.open_image(single_image_name_1)[:, :, [2,1,0]]
                img_as_tensor_1 = self.totensor(img_as_img_1)
                img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_1, img_as_tensor_1], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor
        img_as_tensor_1 = self.normalization(img_as_tensor_1)
        img_as_tensor_2 = self.normalization(img_as_tensor_2)
        img_as_tensor_3 = self.normalization(img_as_tensor_3)

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len


#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 years: Optional[List[int]] = [*range(2000, 2021)],
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': labels,
            'image_ids': selection['image_id'],
            'timestamps': selection['timestamp']
        }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class SentinelOurImageDataset(SatelliteDataset):
    def __init__(self,
                 csv_path: str):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=3)
        self.df = pd.read_csv(csv_path)
        self.tranforms = transforms.Compose([transforms.RandomCrop(224),])

        self.indices = self.df.index.unique().to_numpy()
        self.image_arr = np.asarray(self.df['image_path'])
        self.label_arr = np.asarray(self.df['category'])
        self.data_len = len(self.df)
        self.dataset_root_path = os.path.dirname(csv_path)
        self.name2index = dict(zip([os.path.join(self.dataset_root_path, x) for x in self.image_arr], np.arange(self.data_len)))

        s2_mean = [43.40,47.88,56.28,61.45,71.13,88.79,96.39,98.16,100.87,100.61,94.68,78.27]
        s2_std = [8.10,8.14,8.94,11.39,11.52,13.22,14.98,15.20,15.59,17.59,14.53,13.78]
        l8_mean = [228.25, 234.63, 247.55, 244.49, 253.59, 252.99, 250.60]
        l8_std = [9.62, 8.90, 5.44, 7.30, 1.81, 3.07, 5.34]

        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)
        self.normalization_s2 = SentinelNormalize(s2_mean, s2_std)
        self.normalization_l8 = SentinelNormalize(l8_mean, l8_std)


    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.image_arr[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection)  # (h, w, c)
        if '/l8/' in selection:
            band_seq = random.sample(range(7), k=3)
            images = self.normalization_l8(images)[:,:,band_seq]
        else:
            band_seq = random.sample(range(12), k=3)
            images = self.normalization_s2(images)[:,:,band_seq]

        img_as_tensor = self.scale(self.totensor(images))  # (c, h, w)

        #sample = {
        #    'images': images,
        #    'labels': labels,
        #    'image_ids': idx,
        #    'timestamps': selection['timestamp']
        #}
        return img_as_tensor, self.label_arr[idx]


class DatasetOursTemporalSpectral(SatelliteDataset):
    def __init__(self, csv_path: str):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=6)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info['image_path'])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info['category'])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info['timestamp'])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # hard-coded for fMoW

        mean = [0.07818352, 0.08517814, 0.075134, 0.20921397, 0.1516, 0.09824585]
        std = [0.02141723, 0.02170983, 0.02362014, 0.03475414, 0.02936147, 0.02509896]
        
        self.normalization = SentinelNormalize(mean, std)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]
        
        if 's2ng' in single_image_name_1:
            matched_img = single_image_name_1.replace('s2ng', 's2g')
            if matched_img in self.image_arr:
                temporal_files = [matched_img]
            else:
                temporal_files = []
        else:
            matched_img = single_image_name_1.replace('s2g', 's2ng')
            if matched_img in self.image_arr:
                temporal_files = [matched_img]
            else:
                temporal_files = []

        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]

        img_as_img_1 = self.open_image(single_image_name_1)
        img_as_img_2 = self.open_image(single_image_name_2)
        img_as_img_3 = self.open_image(single_image_name_3)
        img_as_img_1 = self.normalization(img_as_img_1)
        img_as_img_2 = self.normalization(img_as_img_2)
        img_as_img_3 = self.normalization(img_as_img_3)
        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        del img_as_img_3
        img_as_tensor_1 = self.scale(img_as_tensor_1)
        img_as_tensor_2 = self.scale(img_as_tensor_2)
        img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and \
                    img_as_tensor_2.shape[2] > 224 and \
                    img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w],
                    img_as_tensor_2[..., :min_w],
                    img_as_tensor_3[..., :min_w]
                ], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and \
                    img_as_tensor_2.shape[1] > 224 and \
                    img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w, :],
                    img_as_tensor_2[..., :min_w, :],
                    img_as_tensor_3[..., :min_w, :]
                ], dim=-3)
            else:
                img_as_img_1 = self.open_image(single_image_name_1)
                img_as_tensor_1 = self.totensor(img_as_img_1)
                img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_1, img_as_tensor_1], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len

class DatasetOursLanSenTemporalSpectral(SatelliteDataset):
    def __init__(self, csv_path: str):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=6)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info['image_path'])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info['category'])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info['timestamp'])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # hard-coded for fMoW

        mean_sentinel = [0.07818352, 0.08517814, 0.075134, 0.20921397, 0.1516, 0.09824585]
        std_sentinel = [0.02141723, 0.02170983, 0.02362014, 0.03475414, 0.02936147, 0.02509896]
        #  sentinel 'Blue','Green','Red','NIR','SWIR1','SWIR2'
        mean_landsat = [0.02291702, 0.02772821, 0.04881586, 0.04111947, 0.21042395, 0.12758436, 0.07012639]
        std_landsat = [0.02627975, 0.02666088, 0.02772484, 0.03052112, 0.06120895, 0.04521893, 0.03259456]


         # Landsat: std [0.02627975 0.02666088 0.02772484 0.03052112 0.06120895 0.04521893
         # 0.03259456], mean [0.02291702 0.02772821 0.04881586 0.04111947 0.21042395 0.12758436
         #         0.07012639]


        self.normalization_sentinel = SentinelNormalize(mean_sentinel, std_sentinel)
        self.normalization_landsat = SentinelNormalize(mean_landsat, std_landsat)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)

    def open_image(self, img_path, islandsat=False):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)
        if islandsat:
            img = img[1:,:,:]  ##drop 'Coastal','Blue','Green','Red','NIR','SWIR-1','SWIR-2'

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]
        
        if 's2ng' in single_image_name_1:
            temporal_files = {'s2g':single_image_name_1.replace('s2ng', 's2g').replace('S2_n', 'S2'), 'landsat_g':single_image_name_1.replace('s2ng', 'landsat_g').replace('S2_n_10m', 'Landsat_g'), 'landsat_ng':single_image_name_1.replace('s2ng', 'landsat_ng').replace('S2_n_10m', 'Landsat_ng')}
        elif 's2g' in single_image_name_1:
            temporal_files = {'s2ng':single_image_name_1.replace('s2g', 's2ng').replace('S2', 'S2_n'), 'landsat_g':single_image_name_1.replace('s2g', 'landsat_g').replace('S2_10m', 'Landsat_g'), 'landsat_ng':single_image_name_1.replace('s2g', 'landsat_ng').replace('S2_10m', 'Landsat_ng')}
        elif 'landsat_g' in single_image_name_1:
            temporal_files = {'s2ng':single_image_name_1.replace('landsat_g', 's2ng').replace('Landsat_g', 'S2_n_10m'), 's2g':single_image_name_1.replace('landsat_g', 's2g').replace('Landsat_g', 'S2_10m'), 'landsat_ng':single_image_name_1.replace('landsat_g', 'landsat_ng').replace('Landsat_g', 'Landsat_ng')}
        else:
            temporal_files = {'s2ng':single_image_name_1.replace('landsat_ng', 's2ng').replace('Landsat_ng', 'S2_n_10m'), 's2g':single_image_name_1.replace('landsat_ng', 's2g').replace('Landsat_ng', 'S2_10m'), 'landsat_g':single_image_name_1.replace('landsat_ng', 'landsat_g').replace('Landsat_ng', 'Landsat_g')}
        
        
        ## check if any files are not exist
        #for k in temporal_files.keys():
        #    if not os.path.exists(temporal_files[k]):
        #        del temporal_files[k]
        
        a = random.sample(temporal_files.keys(), 1)[0]
        single_image_name_1 = temporal_files[a]
        img_as_img_1 = self.open_image(single_image_name_1)
        del temporal_files[a]
        
        if 'landsat' in a:
            img_as_img_1 = self.normalization_landsat(img_as_img_1)[:,:,1:]
        else:
            img_as_img_1 = self.normalization_sentinel(img_as_img_1)
        
        a = random.sample(temporal_files.keys(), 1)[0]
        single_image_name_2 = temporal_files[a]
        img_as_img_2 = self.open_image(single_image_name_2)
        del temporal_files[a]
        if 'landsat' in a:
            img_as_img_2 = self.normalization_landsat(img_as_img_2)[:,:,1:]
        else:
            img_as_img_2 = self.normalization_sentinel(img_as_img_2)

        a = random.sample(temporal_files.keys(), 1)[0]
        single_image_name_3 = temporal_files[a]
        img_as_img_3 = self.open_image(single_image_name_3)
        del temporal_files[a]
        if 'landsat' in a:
            img_as_img_3 = self.normalization_landsat(img_as_img_3)[:,:,1:]
        else:
            img_as_img_3 = self.normalization_sentinel(img_as_img_3)

        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        del img_as_img_3
        img_as_tensor_1 = self.scale(img_as_tensor_1)
        img_as_tensor_2 = self.scale(img_as_tensor_2)
        img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and \
                    img_as_tensor_2.shape[2] > 224 and \
                    img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w],
                    img_as_tensor_2[..., :min_w],
                    img_as_tensor_3[..., :min_w]
                ], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and \
                    img_as_tensor_2.shape[1] > 224 and \
                    img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w, :],
                    img_as_tensor_2[..., :min_w, :],
                    img_as_tensor_3[..., :min_w, :]
                ], dim=-3)
            else:
                #img_as_img_1 = self.open_image(single_image_name_1)
                #img_as_tensor_1 = self.totensor(img_as_img_1)
                #img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len


class DatasetGrid(SatelliteDataset):
    def __init__(self, csv_path: str, isAnchor=False):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info['image_path'])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info['category'])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)
        self.isAnchor = isAnchor

        self.timestamp_arr = np.asarray(self.data_info['timestamp'])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))
        mean_sentinel = [0.07818352, 0.08517814, 0.075134, 0.20921397, 0.1516, 0.09824585]
        std_sentinel = [0.02141723, 0.02170983, 0.02362014, 0.03475414, 0.02936147, 0.02509896]
        #  sentinel 'Blue','Green','Red','NIR','SWIR1','SWIR2'
        mean_landsat = [0.02291702, 0.02772821, 0.04881586, 0.04111947, 0.21042395, 0.12758436, 0.07012639]
        std_landsat = [0.02627975, 0.02666088, 0.02772484, 0.03052112, 0.06120895, 0.04521893, 0.03259456]

        s2_mean = [43.40,47.88,56.28,61.45,71.13,88.79,96.39,98.16,100.87,100.61,94.68,78.27]
        s2_std = [8.10,8.14,8.94,11.39,11.52,13.22,14.98,15.20,15.59,17.59,14.53,13.78]
        l8_mean = [228.25, 234.63, 247.55, 244.49, 253.59, 252.99, 250.60]
        l8_std = [9.62, 8.90, 5.44, 7.30, 1.81, 3.07, 5.34]


         # Landsat: std [0.02627975 0.02666088 0.02772484 0.03052112 0.06120895 0.04521893
         # 0.03259456], mean [0.02291702 0.02772821 0.04881586 0.04111947 0.21042395 0.12758436
         #         0.07012639]


        self.normalization_sentinel_grid = SentinelNormalize(mean_sentinel, std_sentinel)
        self.normalization_landsat_grid = SentinelNormalize(mean_landsat, std_landsat)
        self.normalization_sentinel_city = SentinelNormalize(s2_mean, s2_std)
        self.normalization_landsat_city = SentinelNormalize(l8_mean, l8_std)

        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)

    def open_image(self, img_path, islandsat=False):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)
        if islandsat:
            img = img[1:,:,:]  ##drop 'Coastal','Blue','Green','Red','NIR','SWIR-1','SWIR-2'

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        images = self.open_image(single_image_name_1)

        if '/l8/' in single_image_name_1:
            band_seq = random.sample(range(images.shape[2]), k=3)
            images = self.normalization_landsat_city(images)[:,:,band_seq]
        elif '/s2/' in single_image_name_1:
            band_seq = random.sample(range(images.shape[2]), k=3)
            images = self.normalization_sentinel_city(images)[:,:,band_seq]

        elif 'landsat_' in single_image_name_1:
            band_seq = random.sample(range(images.shape[2]), k=3)
            images = self.normalization_landsat_grid(images)[:,:,band_seq]               
        else:
            band_seq = random.sample(range(images.shape[2]), k=3)
            images = self.normalization_sentinel_grid(images)[:,:,band_seq]         

        img_as_tensor_1 = self.totensor(images)
        
        if torch.rand(1)>0.5:        
            img_as_tensor_1 = self.scale(img_as_tensor_1)

        img_as_tensor_1 = self.transforms(img_as_tensor_1)

        #ts1 = self.parse_timestamp(single_image_name_1)

        # Get label(class) of the image based on the cropped pandas column
        labels = self.label_arr[index]

        return img_as_tensor_1, labels

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len


class DatasetDYNAnchor(SatelliteDataset):
    def __init__(self, csv_path: str, isAnchor=False):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=6)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info['image_path'])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info['category'])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)
        self.isAnchor = isAnchor

        self.timestamp_arr = np.asarray(self.data_info['timestamp'])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # hard-coded for fMoW

        dny_s2_mean = [1161.52559058, 1399.3856684,  1455.72690441, 2761.06478679, 1815.21453206, 2465.56254884, 2722.33995034, 2867.82050836, 2336.82199658, 1742.14946239, 1069.34656132, 3128.77439469]
        dny_s2_std = [523.98821885, 536.77753748, 625.56680443, 771.04933568, 616.8591111, 671.86245863, 727.05030397, 756.91746641, 683.80194507, 629.27708749, 465.96431735, 848.05289899]
        
        dny_pl_mean = [671.26011478, 915.61776057, 1042.59598012, 2605.19344952]
        dny_pl_std = [261.97106523, 298.38734043, 413.97257355, 596.98879655]


         # Landsat: std [0.02627975 0.02666088 0.02772484 0.03052112 0.06120895 0.04521893
         # 0.03259456], mean [0.02291702 0.02772821 0.04881586 0.04111947 0.21042395 0.12758436
         #         0.07012639]


        self.normalization_s2 = SentinelNormalize(dny_s2_mean, dny_s2_std)
        self.normalization_pl = SentinelNormalize(dny_pl_mean, dny_pl_std)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)  ## to be optimized

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]
        
        anchor_flag = []
        
        if 'planet' in single_image_name_1:
            location_index = single_image_name_1.split('/')[9]
            temporal_files = [i for i in self.image_arr if location_index in i]
            temporal_files.remove(single_image_name_1)
            anchor_flag.append(False)

        elif 'sentinel2' in single_image_name_1:
            location_index = single_image_name_1.split('/')[7]
            temporal_files = [i for i in self.image_arr if location_index in i]
            temporal_files.remove(single_image_name_1)
            anchor_flag.append(True)
            
        img_as_img_1 = self.open_image(single_image_name_1)
        
        if self.isAnchor:
            source_index = ['planet', 'sentinel2']
            source_choice = random.choice(source_index)
            
            if source_choice=='planet':anchor_flag.append(False)
            else:anchor_flag.append(True)
            
            temp_files = [i for i in temporal_files if source_choice in i]
            if len(temp_files)>0:
                single_image_name_2 = random.choice(temp_files)
            else: 
                single_image_name_2 = single_image_name_1
                anchor_flag[1] = True
            img_as_img_2 = self.open_image(single_image_name_2)
            
            source_index.remove(source_choice)
            source_choice = source_index[0]
            if source_choice=='planet':anchor_flag.append(False)
            else:anchor_flag.append(True)
            temp_files = [i for i in temporal_files if source_choice in i]
            if len(temp_files)>0:
                single_image_name_3 = random.choice(temp_files)
            else:
                single_image_name_3 = single_image_name_1
                anchor_flag[2] = True
            img_as_img_3 = self.open_image(single_image_name_3)
            
        else:
            single_image_name_2 = random.choice(temporal_files)     
            temporal_files.remove(single_image_name_2)
            img_as_img_2 = self.open_image(single_image_name_2)
            
            single_image_name_3 = random.choice(temporal_files)
            img_as_img_3 = self.open_image(single_image_name_3)
            

        imglist = [img_as_img_1, img_as_img_2, img_as_img_3]
        
        for i in range(3):
            if imglist[i].shape[2]>4:
                band_seq = random.sample(range(12), k=3)
                imglist[i] = self.normalization_s2(imglist[i])[:,:,band_seq]               
            else:
                band_seq = random.sample(range(4), k=3)
                imglist[i] = self.normalization_pl(imglist[i])[:,:,band_seq] 

        img_as_tensor_1 = self.totensor(imglist[0])
        img_as_tensor_2 = self.totensor(imglist[1])
        img_as_tensor_3 = self.totensor(imglist[2])
        del img_as_img_1,img_as_img_2,img_as_img_3,imglist
        
        if torch.rand(1)>0.5:        
            img_as_tensor_1 = self.scale(img_as_tensor_1)
            img_as_tensor_2 = self.scale(img_as_tensor_2)
            img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and \
                    img_as_tensor_2.shape[2] > 224 and \
                    img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w],
                    img_as_tensor_2[..., :min_w],
                    img_as_tensor_3[..., :min_w]
                ], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and \
                    img_as_tensor_2.shape[1] > 224 and \
                    img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w, :],
                    img_as_tensor_2[..., :min_w, :],
                    img_as_tensor_3[..., :min_w, :]
                ], dim=-3)
            else:
                #img_as_img_1 = self.open_image(single_image_name_1)
                #img_as_tensor_1 = self.totensor(img_as_img_1)
                #img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)
        if self.isAnchor: ts = np.array(anchor_flag)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len


###################################################################################################################
## the class below is the our constructed anchoraware dataloader
# class DatasetAnchorAware(SatelliteDataset):
#     def __init__(self, csv_path: str, isGeoembed=False, isAnchor=False, isScale=False):
#         """
#         Creates temporal dataset for fMoW RGB
#         :param csv_path: Path to csv file containing paths to images
#         :param isgeoembed: bool for whether use/read geoembed information from raw data
#         :param isAnchor: bool for whether use anchor aware module in the dataloader
#         :param isscale: bool for whether use scale embed information in the dataloader
#         """
#         super().__init__(in_c=3)
# 
#         # Transforms
#         self.transforms = transforms.Compose([
#             transforms.Scale(224),
#             #transforms.RandomCrop(224),
#         ])
# 
#         # read from bool function
#         self.isGeoembed = isGeoembed
#         self.isAnchor = isAnchor
#         self.isScale = isScale
#         # Read the csv file
#         self.data_info = pd.read_csv(csv_path)
#         # First column contains the image paths
#         self.image_arr = np.asarray(self.data_info['image_path'])
#         # Second column is the labels
#         # self.label_arr = np.asarray(self.data_info['category'])
# 
#         # Read geoembed information from csv
#         if isGeoembed==True:
#             self.geoembed = np.asarray(self.data_info['embed'])
#         # Calculate len
#         self.data_len = len(self.data_info)
# 
#         self.dataset_root_path = os.path.dirname(csv_path)
# 
#         self.name2index = dict(zip(
#             [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
#             np.arange(self.data_len)
#         ))
# 
# 
#         # calculated mean and std information for each dataset
#         # mean and std for grid dataset
#         mean_sentinel_grid = [0.07818352, 0.08517814, 0.075134, 0.20921397, 0.1516, 0.09824585]
#         std_sentinel_grid = [0.02141723, 0.02170983, 0.02362014, 0.03475414, 0.02936147, 0.02509896]
#         mean_landsat_grid = [0.02291702, 0.02772821, 0.04881586, 0.04111947, 0.21042395, 0.12758436, 0.07012639]
#         std_landsat_grid = [0.02627975, 0.02666088, 0.02772484, 0.03052112, 0.06120895, 0.04521893, 0.03259456]
# 
#         self.normalization_sentinel_grid = SentinelNormalize(mean_sentinel_grid, std_sentinel_grid)
#         self.normalization_landsat_grid = SentinelNormalize(mean_landsat_grid, std_landsat_grid)
# 
#         # mean and std for city dataset (seco-like)
#         city_s2_mean = [43.40,47.88,56.28,61.45,71.13,88.79,96.39,98.16,100.87,100.61,94.68,78.27]
#         city_s2_std = [8.10,8.14,8.94,11.39,11.52,13.22,14.98,15.20,15.59,17.59,14.53,13.78]
#         city_l8_mean = [228.25, 234.63, 247.55, 244.49, 253.59, 252.99, 250.60]
#         city_l8_std = [9.62, 8.90, 5.44, 7.30, 1.81, 3.07, 5.34]
# 
#         self.normalization_sentinel_city = SentinelNormalize(city_s2_mean, city_s2_std)
#         self.normalization_landsat_city = SentinelNormalize(city_l8_mean, city_l8_std)
# 
#         # mean and std for GF1, GF2, and corresponding sentinel2 dataset
#         gf1_mean = [96.11124649, 98.68236718, 99.79782819, 99.677417]
#         gf1_std = [37.65278275, 36.19969664, 35.67627815, 38.7231676]
#         gf2_mean = [78.67503623, 81.1744213 , 85.88216811, 86.31217156]
#         gf2_std = [33.51362863, 33.19806934, 33.76323961, 36.0095568]
#         sentinel_mean_gf1 = [43.40,47.88,56.28,61.45,71.13,88.79,96.39,98.16,100.87,100.61,94.68,78.27] ## need to be modified
#         sentinel_std_gf1 = [8.10,8.14,8.94,11.39,11.52,13.22,14.98,15.20,15.59,17.59,14.53,13.78] ## need to be modified
#         sentinel_mean_gf2 = [43.40,47.88,56.28,61.45,71.13,88.79,96.39,98.16,100.87,100.61,94.68,78.27] ## need to be modified
#         sentinel_std_gf2 = [8.10,8.14,8.94,11.39,11.52,13.22,14.98,15.20,15.59,17.59,14.53,13.78] ## need to be modified
# 
#         self.normalization_gf1 = SentinelNormalize(gf1_mean, gf2_std)
#         self.normalization_gf2 = SentinelNormalize(gf2_mean, gf2_std)
#         self.normalization_sentinel_gf1 = SentinelNormalize(sentinel_mean_gf1, sentinel_std_gf1)
#         self.normalization_sentinel_gf2 = SentinelNormalize(sentinel_mean_gf2, sentinel_std_gf2)
# 
#         self.totensor = transforms.ToTensor()
#         self.scale = transforms.Resize(224)  ## to be optimized
# 
#     def open_image(self, img_path):
#         with rasterio.open(img_path) as data:
#             img = data.read()  # (c, h, w)
#             res = data.res[0]  # res refer to the spatial resolution of the given image, which is supposed to be kept in the lmdb metadata
# 
#         return img.transpose(1, 2, 0).astype(np.float32), res  # (h, w, c)
# 
#     def __getitem__(self, index):
#         # Get image name from the pandas df
#         single_image_name_1 = self.image_arr[index]
# 
#         anchor_flag = []
# 
#         #########################################################################################################################################################
#         ###  The codes below are used to generate image list 'imglist', which is consist of three randomly selected image ###
#         ###  The generated 'imglist' should like imglist = [img1, img2, img3], each item in [img1, img2, img3] is an absolute path of image. ###
#         ###  This codes are supposed to be removed in the lmdb versions
# 
#         if 'planet' in single_image_name_1:
#             location_index = single_image_name_1.split('/')[9]
#             temporal_files = [i for i in self.image_arr if location_index in i]
#             temporal_files.remove(single_image_name_1)
#             anchor_flag.append(False)
# 
#         elif 'sentinel2' in single_image_name_1:
#             location_index = single_image_name_1.split('/')[7]
#             temporal_files = [i for i in self.image_arr if location_index in i]
#             temporal_files.remove(single_image_name_1)
#             anchor_flag.append(True)
# 
#         if self.isAnchor:
#             source_index = ['planet', 'sentinel2']
#             source_choice = random.choice(source_index)
# 
#             if source_choice=='planet':anchor_flag.append(False)
#             else:anchor_flag.append(True)
# 
#             temp_files = [i for i in temporal_files if source_choice in i]
#             if len(temp_files)>0:
#                 single_image_name_2 = random.choice(temp_files)
#             else:
#                 single_image_name_2 = single_image_name_1
#                 anchor_flag[1] = True
# 
#             source_index.remove(source_choice)
#             source_choice = source_index[0]
#             if source_choice=='planet':anchor_flag.append(False)
#             else:anchor_flag.append(True)
#             temp_files = [i for i in temporal_files if source_choice in i]
#             if len(temp_files)>0:
#                 single_image_name_3 = random.choice(temp_files)
#             else:
#                 single_image_name_3 = single_image_name_1
#                 anchor_flag[2] = True
# 
#         else:
#             single_image_name_2 = random.choice(temporal_files)
#             temporal_files.remove(single_image_name_2)
# 
#             single_image_name_3 = random.choice(temporal_files)
# 
# 
#         imglist = [single_image_name_1, single_image_name_2, single_image_name_3]
# 
#         ### The marked codes stop here ###
#         ##################################################################################################################
# 
#         imgs_data = imglist
#         imgs_res = imglist
#         for i in range(len(imglist)):
#             _data, _res = self.open_image(imglist[i])
#             imgs_res[i] = _res
#             bands_length = _data.shape[-1]
#             band_seq = random.sample(range(bands_length), k=3)
# 
#             ##############################################################################################################
#             ### The codes below are used to identify which normalization parameters are supposed to apply
#             ### This codes are supposed to be removed in the lmdb version
#             if '/l8/' in imglist[i]:
#                 imgs_data[i] = self.normalization_landsat_city(_data)[:,:,band_seq]
#             elif '/s2/' in imglist[i]:
#                 imgs_data[i] = self.normalization_sentinel_city(_data)[:,:,band_seq]
#             elif '/GF1/' in imglist[i]:
#                 imgs_data[i] = self.normalization_gf1(_data)[:,:,band_seq]
#             elif '/GF2/' in imglist[i]:
#                 imgs_data[i] = self.normalization_gf2(_data)[:,:,band_seq]
#             elif '/crop_by_GF1_resize/' in imglist[i]:
#                 imgs_data[i] = self.normalization_sentinel_gf1(_data)[:,:,band_seq]
#             elif '/crop_by_GF2_resize/' in imglist[i]:
#                 imgs_data[i] = self.normalization_sentinel_gf2(_data)[:,:,band_seq]
#             elif 'landsat_' in imglist[i]:
#                 imgs_data[i] = self.normalization_landsat_grid(_data)[:,:,band_seq]
#             else:
#                 imgs_data[i] = self.normalization_sentinel_grid(_data)[:,:,band_seq]
#             ### The marked codes stop here
#             ###############################################################################################################
# 
#         img_as_tensor_1 = self.totensor(imgs_data[0])
#         img_as_tensor_2 = self.totensor(imgs_data[1])
#         img_as_tensor_3 = self.totensor(imgs_data[2])
#         del imgs_data
# 
#         if torch.rand(1)>0.5:
#             img_as_tensor_1 = self.scale(img_as_tensor_1)
#             img_as_tensor_2 = self.scale(img_as_tensor_2)
#             img_as_tensor_3 = self.scale(img_as_tensor_3)
#         try:
#             if img_as_tensor_1.shape[2] > 224 and \
#                     img_as_tensor_2.shape[2] > 224 and \
#                     img_as_tensor_3.shape[2] > 224:
#                 min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
#                 img_as_tensor = torch.cat([
#                     img_as_tensor_1[..., :min_w],
# 
#         try:
#             if img_as_tensor_1.shape[2] > 224 and \
#                     img_as_tensor_2.shape[2] > 224 and \
#                     img_as_tensor_3.shape[2] > 224:
#                 min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
#                 img_as_tensor = torch.cat([
#                     img_as_tensor_1[..., :min_w],
#                     img_as_tensor_2[..., :min_w],
#                     img_as_tensor_3[..., :min_w]
#                 ], dim=-3)
#             elif img_as_tensor_1.shape[1] > 224 and \
#                     img_as_tensor_2.shape[1] > 224 and \
#                     img_as_tensor_3.shape[1] > 224:
#                 min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
#                 img_as_tensor = torch.cat([
#                     img_as_tensor_1[..., :min_w, :],
#                     img_as_tensor_2[..., :min_w, :],
#                     img_as_tensor_3[..., :min_w, :]
#                 ], dim=-3)
#             else:
#                 #img_as_img_1 = self.open_image(single_image_name_1)
#                 #img_as_tensor_1 = self.totensor(img_as_img_1)
#                 #img_as_tensor_1 = self.scale(img_as_tensor_1)
#                 img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=-3)
#         except:
#             print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
#             assert False
# 
#         del img_as_tensor_1
#         del img_as_tensor_2
#         del img_as_tensor_3
# 
#         img_as_tensor = self.transforms(img_as_tensor)
#         img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
#         del img_as_tensor
# 
#         ### ge is the geo-embedding information, which is a list containing three array, e.g. ge = [array1, array2, array3]
#         if self.isGeoembed:
#             ge1 = self.parse_geoembed(imglist[0])
#             ge2 = self.parse_geoembed(imglist[1])
#             ge3 = self.parse_geoembed(imglist[2])
#             ge = [ge1, ge2, ge3]
# 
#         ## res is the spatial resolution information, which is a tensor, e.g. res = tensor([res1, res2, res3])
#         res = torch.tensor(imgs_res)
# 
#         ### anchor is a list containing one 'True' and two 'False', each item in achor is a bool value, e.g., anchor = [True, False, False]
#         if self.isAnchor: anchor = np.array(anchor_flag)
# 
#         imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)
# 
#         del img_as_tensor_1
#         del img_as_tensor_2
#         del img_as_tensor_3
# 
#         return (imgs, ge, anchor, res)
# 
#     def parse_timestamp(self, name):
#         timestamp = self.timestamp_arr[self.name2index[name]]
#         year = int(timestamp[:4])
#         month = int(timestamp[5:7])
#         hour = int(timestamp[11:13])
#         return np.array([year - self.min_year, month - 1, hour])
# 
#     def parse_geoembed(self, name):
#         embed = self.embed[self.name2index[name]]
#         return embed
# 
#     def __len__(self):
# 
#         return self.data_len







class EuroSat(SatelliteDataset):
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

    def __init__(self, file_path, transform, masked_bands=None, dropped_bands=None):
        """
        Creates dataset for multi-spectral single image classification for EuroSAT.
        :param file_path: path to txt file containing paths to image data for EuroSAT.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(13)
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        self.img_paths = [row.split()[0] for row in data]
        self.labels = [int(row.split()[1]) for row in data]

        self.transform = transform

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.img_paths)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        if self.masked_bands is not None:
            img[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        img_as_tensor = self.transform(img)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return img_as_tensor, label

class DatasetLMDB(SatelliteDataset):

    def __init__(self, lmdb_path: str, isGeoembed=False, isAnchor=False, isScale=False):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        self.data_type = "lmdb"
        self.lmdb_path = lmdb_path
        self.lmdb_env = self._init_lmdb()
        self.keys_path = "/share/home/thuzjx/data/city_metainfo.pkl"
        assert self.lmdb_path, "Error: lmdb_path is Empty!!" 
        self.keys = self._get_image_keys_from_lmdb()
        self.data_len = len(self.keys)

        self.isGeoembed = isGeoembed
        self.isAnchor = isAnchor
        self.isScale = isScale

        self.totensor = transforms.ToTensor()

    def _init_lmdb(self):
        self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def _get_image_keys_from_lmdb(self):
        keys_path = self.keys_path
        meta_info = pickle.load(open(keys_path, "rb"))
        keys = meta_info['keys']
	
        return keys

    def _read_img_lmdb(self, currentkey, size=(3, 3, 224, 224) ):
        env = self.lmdb_env
        key = currentkey 
        with env.begin(write=False) as txn:
            buf = txn.get( key.encode() )
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        T,C,H,W = size
        img = img_flat.reshape(T, H, W, C)
        # img.flags.writeable = True
        img =img.copy()
        return img    

    def __getitem__(self, index):
        if self.data_type=='lmdb' and (self.lmdb_env is None):
            self._init_lmdb()
        current_key = self.keys[index]
        images = self._read_img_lmdb(currentkey=current_key)
        # images = copy.deepcopy(images_readonly)
        
        img_as_tensor_1 = torch.stack((self.totensor(images[0,:,:,:]), self.totensor(images[1,:,:,:]), self.totensor(images[2,:,:,:]) ),   dim=0 )#  3,3,224,224
        # ts1 = np.array([0, 5, 9])
        # ts = np.stack([ts1, ts1, ts1], axis=0)
        ge = []
        if self.isGeoembed:
            # current_key.split('_')[4] 5 6 7     8 9 10 11
            ge1 = self.geo_int2code(current_key.split('_')[4:12]) 
            ge2 = self.geo_int2code(current_key.split('_')[12:20]) 
            ge3 = self.geo_int2code(current_key.split('_')[20:28]) 
            ge = [ge1, ge2, ge3]

        anchor = [False, False, False] 
        if self.isAnchor:
            anchor[random.randint(0,2)] = True

        anchor = torch.tensor(anchor)

        res = []
        if self.isScale:
            for i in range(1,4):
                res.append(int(current_key.split('_')[i]))
        res = torch.tensor(res)
        
        return (img_as_tensor_1, ge, anchor ,res)
        # return img_as_tensor_1, torch.tensor(ts), 0

    def __len__(self):
        return self.data_len

    def geo_int2code(self, cord_list):
        s = ''
        for i in cord_list:
            s += "{0:0=32b}".format(int(i))
        return s



def build_fmow_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    csv_path = os.path.join(args.train_path if is_train else args.test_path)
    if args.dataset_type == 'city_lmdb':
        dataset = DatasetLMDB(lmdb_path="/share/home/thuzjx/data/batch1_city1.lmdb") 
    elif args.dataset_type == 'city_lmdb_full':
        dataset = DatasetLMDB(lmdb_path='/share/home/thuzjx/data/city.lmdb', isGeoembed=True, isAnchor=True, isScale=True)
    elif args.dataset_type == 'rgb':
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = CustomDatasetFromImages.build_transform(is_train, args.input_size, mean, std)
        dataset = CustomDatasetFromImages(csv_path, transform)
    elif args.dataset_type == 'temporal':
        dataset = CustomDatasetFromImagesTemporal(csv_path)
    elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(csv_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)
   
    elif args.dataset_type == 'city150k':
        dataset = SentinelOurImageDataset(csv_path)
    elif args.dataset_type == 'temporal_ours':
        dataset = CustomDatasetFromOursTemporal(csv_path)
    elif args.dataset_type == 'sentinel_ours':
        mean = [0.07818352, 0.08517814, 0.075134, 0.20921397, 0.1516, 0.09824585]
        std = [0.02141723, 0.02170983, 0.02362014, 0.03475414, 0.02936147, 0.02509896]
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelOurImageDataset(csv_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)
    elif args.dataset_type == 'temporal_spectral_ours':
        dataset = DatasetOursTemporalSpectral(csv_path)
    elif args.dataset_type == 'temporal_spectral_lansen_ours':
        dataset = DatasetOursLanSenTemporalSpectral(csv_path)
    elif args.dataset_type == 'grid_baseline':
        dataset = DatasetGrid(csv_path)
    elif args.dataset_type == 'dynamic_baseline':
        dataset = DatasetDYNAnchor(csv_path)
    elif args.dataset_type == 'dynamic_anchor':
        dataset = DatasetDYNAnchor(csv_path, isAnchor=True)
    elif args.dataset_type == 'rgb_temporal_stacked':
        mean = FMoWTemporalStacked.mean
        std = FMoWTemporalStacked.std
        transform = FMoWTemporalStacked.build_transform(is_train, args.input_size, mean, std)
        dataset = FMoWTemporalStacked(csv_path, transform)
    elif args.dataset_type == 'euro_sat':
        mean, std = EuroSat.mean, EuroSat.std
        transform = EuroSat.build_transform(is_train, args.input_size, mean, std)
        dataset = EuroSat(csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands)
    elif args.dataset_type == 'naip':
        from util.naip_loader import NAIP_train_dataset, NAIP_test_dataset, NAIP_CLASS_NUM
        dataset = NAIP_train_dataset if is_train else NAIP_test_dataset
        args.nb_classes = NAIP_CLASS_NUM
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset
