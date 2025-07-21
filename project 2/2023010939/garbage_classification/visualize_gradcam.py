import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append("/home/stu4/garbage_classification")
from data_loader import load_data
from model import BasicCNN, ResNetModel, ImprovedResNet
from train import train_model
from evaluate import evaluate_model
from utils import set_seed, plot_training_history, count_parameters, visualize_model_predictions
from visualization import visualize_gradcam

def visualize_gradcam_main(model_path, data_dir='garbage-dataset', batch_size=16, img_size=224, data_augmentation=False, seed=42, model_type='basic'):
    # 设置随机种子
    set_seed(seed)
    
    # 检查可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, class_names = load_data(
        data_dir,
        batch_size=batch_size,
        img_size=img_size,
        augment=data_augmentation
    )
    
    # 创建模型
    if model_type == 'basic':
        model = BasicCNN(num_classes=len(class_names))
    elif model_type == 'resnet':
        model = ResNetModel(num_classes=len(class_names), pretrained=True)
    elif model_type == 'improved':
        model = ImprovedResNet(num_classes=len(class_names), pretrained=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = model.to(device)
    
    # 加载模型参数
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 使用Grad-CAM可视化模型关注区域
    visualize_gradcam(model, test_loader, device, class_names)
    
if __name__ == "__main__":
    visualize_gradcam_main(
        model_path='best_model.pth',
        data_dir='garbage-dataset',
        batch_size=16,
        img_size=224,
        data_augmentation=False,
        seed=42
    )

