import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os

def set_seed(seed=42):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_training_history(history, save_path='training_history.png'):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录字典
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 参数总数
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def visualize_model_predictions(model, test_loader, device, class_names, num_images=6):
    """
    可视化模型预测结果
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        class_names: 类别名称列表
        num_images: 显示的图像数量
    """
    model.eval()
    
    # 获取一批数据
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # 显示图像和预测结果
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(2, 3, i + 1)
        # 反归一化图像
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f'Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.close()