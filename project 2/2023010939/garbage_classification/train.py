import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=200, scheduler=None, save_dir='checkpoints', patience=20, min_delta=0.001):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        scheduler: 学习率调度器
        save_dir: 模型保存目录
        patience: 早停轮数
        min_delta: 最小变化量
        
    Returns:
        model: 训练好的模型
        history: 训练历史记录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 初始化训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 记录最佳验证准确率
    best_val_acc = 0.0
    early_stopping_count = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        # tqdm显示进度条
        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}')
        
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
            
            # 统计损失和准确率
            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data).item()
            
            train_loss += batch_loss
            train_corrects += batch_corrects
            
            # 更新进度条
            train_pbar.set_postfix({'loss': batch_loss / inputs.size(0), 
                                   'acc': batch_corrects / inputs.size(0)})
        
        # 计算训练集上的平均损失和准确率
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_corrects / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        # 使用tqdm显示进度条
        val_pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}')
        
        for inputs, labels in val_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data).item()
            
            val_loss += batch_loss
            val_corrects += batch_corrects
            
            # 更新进度条
            val_pbar.set_postfix({'loss': batch_loss / inputs.size(0), 
                                 'acc': batch_corrects / inputs.size(0)})
        
        # 计算验证集上的平均损失和准确率
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects / len(val_loader.dataset)
        
        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()
        
        # 打印训练信息
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        
        # 保存训练历史记录
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # 如果当前模型是最佳模型，保存模型
        if epoch_val_acc > best_val_acc + min_delta:
            best_val_acc = epoch_val_acc
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
            }, model_path)
            print(f'Saved best model with val_acc: {epoch_val_acc:.4f}')
        
        else:
            early_stopping_count += 1
            if early_stopping_count >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with val_acc: {checkpoint['val_acc']:.4f}")
    
    return model, history