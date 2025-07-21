import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import seaborn as sns

def evaluate_model(model, test_loader, criterion, device, class_names):
    """
    评估模型
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备
        class_names: 类别名称列表
        
    Returns:
        results: 评估结果字典
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    # 在测试集上进行预测
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            test_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失
    test_loss = test_loss / len(test_loader.dataset)
    
    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算每个类别的精确率、召回率和F1分数
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    # 计算宏平均和加权平均的精确率、召回率和F1分数
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 可视化每个类别的精确率、召回率和F1分数
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Precision, Recall and F1-score for each class')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('class_metrics.png')
    plt.close()
    
    # 保存所有评估结果
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'class_precision': precision,
        'class_recall': recall,
        'class_f1': f1,
        'class_support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    return results