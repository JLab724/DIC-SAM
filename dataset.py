import torchvision.transforms.functional as F
import numpy as np
import random
import os
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 图像转换为tensor并归一化到[0,1]
        image_tensor = F.to_tensor(image)
        # 标签转换为tensor，保持为类别索引（long类型）
        label_array = np.array(label)
        label_tensor = torch.from_numpy(label_array).long()
        return {'image': image_tensor, 'label': label_tensor}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 标签使用最近邻插值，避免产生新的类别值
        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.620, 0.639, 0.594], std=[0.245, 0.219, 0.281]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # 标签是RGB格式，像素值为0,1,2，转换为单通道类别索引
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                # RGB图像，取第一个通道（假设三个通道值相同）
                label = img_array[:, :, 0]  # 取R通道
            else:
                label = img_array
            # 确保标签值在[0, 2]范围内
            label = np.clip(label, 0, 2).astype(np.uint8)
            return Image.fromarray(label, mode='L')


class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # 标签是RGB格式，像素值为0,1,2，转换为单通道类别索引
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                # RGB图像，取第一个通道（假设三个通道值相同）
                label = img_array[:, :, 0]  # 取R通道
            else:
                label = img_array
            # 确保标签值在[0, 2]范围内
            label = np.clip(label, 0, 2).astype(np.uint8)
            return Image.fromarray(label, mode='L')