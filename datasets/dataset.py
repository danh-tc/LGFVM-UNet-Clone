import numpy as np
import os

import random
import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
import cv2
    
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0,1))
    label = np.rot90(label, k, axes=(0,1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis)
    label = np.flip(label, axis=axis)
    return image.copy(), label.copy()

def random_rotate(image, label):
    angle = np.random.uniform(-25, 25)
    rotated_channels = []
    for c in range(image.shape[-1]):
        rotated_c = ndimage.rotate(
            image[..., c], angle, order=1, 
            reshape=False, mode='nearest'
        )
        rotated_channels.append(rotated_c)
    image_rot = np.stack(rotated_channels, axis=-1)
    label_rot = ndimage.rotate(
        label, angle, order=0, 
        reshape=False, mode='nearest'
    )
    return image_rot, label_rot

def elastic_transform(image, label, alpha=1000, sigma=30):
    shape = image.shape[:2] 
    random_state = np.random.RandomState(None)
    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), 
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), 
        sigma, mode="constant"
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    dist_image = []
    for c in range(image.shape[-1]):
        channel = map_coordinates(
            image[..., c], indices, 
            order=3, mode='reflect'
        ).reshape(shape)
        dist_image.append(channel)
    dist_image = np.stack(dist_image, axis=-1)
    dist_label = map_coordinates(
        label, indices, 
        order=0, mode='reflect'
    ).reshape(shape)
    
    return dist_image, dist_label


class AdvancedMedicalAug(torch.nn.Module):
    def __init__(self, aug_prob=0.8):
        super().__init__()
        self.aug_prob = aug_prob 
        
    def forward(self, image, label):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        if random.random() < self.aug_prob:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            else:
                image, label = random_rotate(image, label)
 
        if random.random() < 0.3:
            image, label = elastic_transform(image, label)
        
        image = self.intensity_augment(image)
        return image, label
    
    def intensity_augment(self, image):

        for c in range(image.shape[-1]):
            image[..., c] = np.clip(
                image[..., c] * random.uniform(0.7, 1.3), 
                0, 1
            )
            if random.random() < 0.2:
                noise = np.random.normal(0, 0.05, image[..., c].shape)
                image[..., c] = np.clip(image[..., c] + noise, 0, 1)
        return image


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, transform=None):
        self.transform = transform 
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name)
        data = np.load(data_path)
        image = data['data'].astype(np.float32)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if 'label' in data.files:
            label = data['label'].astype(np.int32)
        else:
            label = np.zeros_like(image, dtype=np.int32)

        if image.shape != self.img_size:
            x, y, z = image.shape 

            image = zoom(image, (self.img_size[0] / x, self.img_size[1] / y, z), order=3)  
            label = zoom(label, (self.img_size[0] / x, self.img_size[1] / y), order=0)

        if self.transform:
            image, label = self.transform(image, label)
        if self.split == 'train':
            image, label = torch.from_numpy(image), torch.from_numpy(label)
        sample = {'image': image, 'label': label, 'case_name': slice_name}
        return sample



