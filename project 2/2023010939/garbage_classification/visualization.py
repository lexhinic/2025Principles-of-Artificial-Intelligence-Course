import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

def apply_gradcam(model, img_tensor, target_layer, device, class_idx=None):
    """
    应用Grad-CAM算法可视化模型关注区域
    
    Args:
        model: 训练好的模型
        img_tensor: 输入图像张量
        target_layer: 目标层
        device: 计算设备
        class_idx: 类别索引
        
    Returns:
        visualization: 可视化结果
    """
    model.eval()
    
    # 创建GradCAM对象
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # 创建目标类别
    if class_idx is not None:
        def target_category(x):
            if len(x.shape) == 1:
                # 一维张量 
                return x[class_idx]
            else:
                # 二维或更高维张量 
                return x[:, class_idx]
        
        targets = [target_category]
    else:
        targets = None
    
    # 获取热力图
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(device), targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 将图像转换为numpy数组
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # 将热力图叠加到原图上
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
    return visualization

def visualize_gradcam(model, test_loader, device, class_names, num_images=4):
    """
    使用Grad-CAM可视化模型的关注区域
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        class_names: 类别名称列表
        num_images: 显示的图像数量
    """
    # 获取一批数据
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # 确定目标层
    if hasattr(model, 'resnet'):
        target_layer = model.resnet.layer4[-1]
    else:
        target_layer = model.features[-3]
    
    # 获取预测结果
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # 为每个图像生成Grad-CAM可视化
    plt.figure(figsize=(12, 3 * num_images))
    for i in range(num_images):
        # 原始图像
        plt.subplot(num_images, 3, 3*i+1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f'Original: {class_names[labels[i]]}')
        plt.axis('off')
        
        # 预测类别的Grad-CAM
        plt.subplot(num_images, 3, 3*i+2)
        pred_cam = apply_gradcam(model, images[i], target_layer, device, preds[i].item())
        plt.imshow(pred_cam)
        plt.title(f'Pred: {class_names[preds[i]]}')
        plt.axis('off')
        
        # 真实类别的Grad-CAM
        plt.subplot(num_images, 3, 3*i+3)
        true_cam = apply_gradcam(model, images[i], target_layer, device, labels[i].item())
        plt.imshow(true_cam)
        plt.title(f'True: {class_names[labels[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualizations.png')
    plt.close()