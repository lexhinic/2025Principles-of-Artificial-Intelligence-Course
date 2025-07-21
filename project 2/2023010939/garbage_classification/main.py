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

def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, class_names = load_data(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=args.data_augmentation
    )
    
    # 创建模型
    if args.model_type == 'basic':
        model = BasicCNN(num_classes=len(class_names))
    elif args.model_type == 'resnet':
        model = ResNetModel(num_classes=len(class_names), pretrained=True)
    elif args.model_type == 'improved':
        model = ImprovedResNet(num_classes=len(class_names), pretrained=True)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 打印模型参数统计信息
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                             momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                              weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
    
    # 训练模型
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device,
        num_epochs=args.epochs,
        scheduler=scheduler,
        save_dir=args.save_dir
    )
    
    # 绘制训练历史曲线
    plot_training_history(history)
    
    # 在测试集上评估模型
    results = evaluate_model(model, test_loader, criterion, device, class_names)
    
    # 打印评估结果
    print("\nTest Results:")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted Precision: {results['weighted_precision']:.4f}")
    print(f"Weighted Recall: {results['weighted_recall']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    
    # 可视化模型预测
    visualize_model_predictions(model, test_loader, device, class_names)
    
    # 使用Grad-CAM可视化模型关注区域
    visualize_gradcam(model, test_loader, device, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Garbage Classification')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./garbage_classification', 
                        help='数据集根目录')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批次大小')
    parser.add_argument('--data_augmentation', action='store_true', 
                        help='是否使用数据增强')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='resnet',
                        choices=['basic', 'resnet', 'improved'],
                        help='模型类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20, 
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='优化器类型')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                        help='模型保存目录')
    
    args = parser.parse_args()
    main(args)