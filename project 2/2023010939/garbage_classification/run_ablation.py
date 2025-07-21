import torch
import os
import numpy as np
from datetime import datetime
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from data_loader import load_data
from ablation_models import BasicCNN_NoDrop, BasicCNN_NoBN, BasicCNN_NoRegularization
from model import BasicCNN
from utils import set_seed
from train import train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 设置随机种子
    set_seed(42)
    
    # 配置参数
    data_dir = "./garbage-dataset"
    batch_size = 16
    img_size = 224
    learning_rate = 0.0001
    weight_decay = 0.0001
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    train_loader, val_loader, test_loader, class_names = load_data(
        data_dir, batch_size, img_size, augment=False
    )
    
    # 创建实验结果目录
    results_dir = f"./ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 定义模型列表及其名称
    models = [
        ("BasicCNN", BasicCNN(num_classes=len(class_names))),
        ("BasicCNN_wo_Dropout", BasicCNN_NoDrop(num_classes=len(class_names))),
        ("BasicCNN_wo_BatchNorm", BasicCNN_NoBN(num_classes=len(class_names))),
        ("BasicCNN_wo_Regularization", BasicCNN_NoRegularization(num_classes=len(class_names)))
    ]
    
    # 保存实验结果
    results = {}
    
    # 运行每个模型
    for model_name, model in models:
        print(f"\n{'='*50}\n开始训练模型: {model_name}\n{'='*50}")
        
        # 优化器和损失函数
        model = model.to(device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 训练模型
        model_save_path = os.path.join(results_dir, f"{model_name.replace(' ', '_')}.pth")
        _, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            device, num_epochs=num_epochs, save_dir=results_dir
        )
        
        # 评估模型
        eval_results = evaluate_model(model, test_loader, criterion, device, class_names)
        
        # 保存结果
        results[model_name] = {
            "accuracy": eval_results["accuracy"],
            "macro_f1": eval_results["macro_f1"],
            "weighted_f1": eval_results["weighted_f1"],
            "val_loss": np.min(history["val_loss"]),
            "train_loss": history["train_loss"][np.argmin(history["val_loss"])]
        }
        
        # 保存混淆矩阵和分类报告
        cm_path = os.path.join(results_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            eval_results["confusion_matrix"], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
    
    # 比较所有模型结果
    print("\n\n==== 消融实验结果汇总 ====")
    print(f"{'模型名称':<25} {'准确率':<10} {'宏平均F1':<10} {'加权平均F1':<10} {'验证损失':<10} {'训练损失':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<25} {result['accuracy']:.4f}    {result['macro_f1']:.4f}    {result['weighted_f1']:.4f}    {result['val_loss']:.4f}     {result['train_loss']:.4f}")
    
    # 可视化结果对比
    plot_metrics = ["accuracy", "macro_f1", "weighted_f1"]
    model_names = list(results.keys())
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(plot_metrics):
        values = [results[model][metric] for model in model_names]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Performance Metrics')
    plt.title('Comparison of Model Variants')
    plt.xticks(x + width, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "performance_comparison.png"))
    plt.close()

if __name__ == "__main__":
    main()