import cv2
import torch
import numpy as np
import random
import os
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
 
class GenerateTargets(object):
    """
    根据增强后的 Mask (0=背景, 1=叶片, 2=病害)，动态生成所有辅助训练任务的 Ground Truth
    """
    def __init__(self, num_grade_levels=6):
        self.num_grade_levels = num_grade_levels
        # 你的阈值: [0%, 5%, 10%, 20%, 50%, 100%]
        self.grade_bounds = [0.001, 0.05, 0.10, 0.20, 0.50, 1.0]

    def _get_grade_coral(self, disease_ratio: float) -> np.ndarray:
        """根据比例计算等级 index，并转换为 CORAL 多热码"""
        grade_idx = len(self.grade_bounds) - 1
        for i, bound in enumerate(self.grade_bounds):
            if disease_ratio <= bound:
                grade_idx = i
                break
                
        # 转换为 K-1 维的 CORAL 标签
        coral_label = np.zeros(self.num_grade_levels - 1, dtype=np.float32)
        if grade_idx > 0:
            coral_label[:grade_idx] = 1.0
        return coral_label

    def __call__(self, data):
        image, mask = data['image'], data['label'] # 此时 mask 还是 PIL Image 或 numpy array
        mask_np = np.array(mask).astype(np.uint8)
        
        # 1. 计算 Grade (病害占比 = 病害面积 / (叶片面积 + 病害面积))
        leaf_pixels = np.sum(mask_np == 1)
        disease_pixels = np.sum(mask_np == 2)
        total_leaf_area = leaf_pixels + disease_pixels
        
        disease_ratio = 0.0
        if total_leaf_area > 0:
            disease_ratio = disease_pixels / total_leaf_area
            
        grade_coral = self._get_grade_coral(disease_ratio)
        
        # 2. 生成 Boundary (边界) 和 Inside (内部)
        # 使用形态学梯度提取病害区域的边界
        disease_mask = (mask_np == 2).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # 膨胀 - 腐蚀 = 边界
        dilated = cv2.dilate(disease_mask, kernel, iterations=1)
        eroded = cv2.erode(disease_mask, kernel, iterations=1)
        boundary_np = dilated - eroded
        inside_np = eroded # 腐蚀后的核心区域作为 inside
        
        # 3. 生成 Leaf Proxy (叶片整体区域，包含病害)
        leaf_proxy_np = (mask_np > 0).astype(np.uint8)

        # 转换为 Tensor
        image_tensor = F.to_tensor(image) # 自动转为 [C, H, W] 并归一化到 0-1
        label_tensor = torch.from_numpy(mask_np).long()
        
        # 将新生成的 Target 打包返回
        return {
            'image': image_tensor,
            'targets': {
                'mask': label_tensor,
                'grade': torch.from_numpy(grade_coral).float(),      # [K-1]
                'boundary': torch.from_numpy(boundary_np).float().unsqueeze(0), # [1, H, W]
                'inside': torch.from_numpy(inside_np).float().unsqueeze(0),     # [1, H, W]
                'leaf_proxy': torch.from_numpy(leaf_proxy_np).float().unsqueeze(0) # [1, H, W]
            }
        }

# ==========================================
# 2. 几何变换类 (保持不变，只做几何操作)
# ==========================================
class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if random.random() < self.p:
            return {'image': F.hflip(data['image']), 'label': F.hflip(data['label'])}
        return data

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if random.random() < self.p:
            return {'image': F.vflip(data['image']), 'label': F.vflip(data['label'])}
        return data

class Normalize(object):
    def __init__(self, mean=[0.620, 0.639, 0.594], std=[0.245, 0.219, 0.281]):
        self.mean = mean
        self.std = std
    def __call__(self, data):
        # 注意这里 data 的结构被 GenerateTargets 改变了
        image = F.normalize(data['image'], self.mean, self.std)
        data['image'] = image
        return data

# ==========================================
# 3. Dataset 类集成
# ==========================================
class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(('.jpg', '.png'))])
        
        # 共享的大豆数据集均值和方差 (务必保证 Train 和 Test 一致)
        soybean_mean = [0.620, 0.639, 0.594]
        soybean_std = [0.245, 0.219, 0.281]
        
        from torchvision import transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                GenerateTargets(num_grade_levels=6), # 核心：放在几何变换之后，ToTensor 之前
                Normalize(mean=soybean_mean, std=soybean_std)
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                GenerateTargets(num_grade_levels=6),
                Normalize(mean=soybean_mean, std=soybean_std)
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data) 
        
        # 返回格式：image tensor, target dictionary
        return data['image'], data['targets']

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            print(f"Loading image: {path}")
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img_array = np.array(Image.open(f))
            if len(img_array.shape) == 3:
                label = img_array[:, :, 0]
            else:
                label = img_array
            label = np.clip(label, 0, 2).astype(np.uint8)
            return Image.fromarray(label, mode='L')