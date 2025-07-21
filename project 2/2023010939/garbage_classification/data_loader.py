import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class GarbageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_data(data_dir, batch_size=32, img_size=224, augment=False):
    """
    加载垃圾分类数据集
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批次大小
        img_size: 图像尺寸
        augment: 是否使用数据增强
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # 定义类别
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    # 收集所有图像路径和对应标签
    all_image_paths = []
    all_labels = []
    
    for cls_name in class_names:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
            
        cls_idx = class_to_idx[cls_name]
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(img_path)
                all_labels.append(cls_idx)
    
    # 划分训练集、验证集和测试集 (70%, 15%, 15%)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_paths, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )
    
    # 基础转换
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 数据增强转换
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = base_transform
    
    # 创建数据集
    train_dataset = GarbageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = GarbageDataset(val_paths, val_labels, transform=base_transform)
    test_dataset = GarbageDataset(test_paths, test_labels, transform=base_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_names