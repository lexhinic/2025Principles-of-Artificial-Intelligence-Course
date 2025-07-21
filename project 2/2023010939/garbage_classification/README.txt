# 垃圾分类项目

本项目实现了一个基于深度学习的垃圾分类系统，能够对10类常见垃圾（如衣服、电池和鞋子等）进行分类。

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- Pillow
- opencv-python
- pytorch-grad-cam

可以使用以下命令安装所需依赖：

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm Pillow opencv-python pytorch-grad-cam
```
或者
```bash
pip install -r requirements.txt
```

## 项目结构
main.py: 主程序入口
data_loader.py: 数据加载和预处理
model.py: 模型定义
train.py: 训练过程
evaluate.py: 模型评估
utils.py: 辅助工具函数
visualize.py: 可视化方法
ablation_models.py: 消融实验模型定义
run_ablation.py: 消融实验脚本
visualize_gradcam.py: Grad-CAM可视化脚本(直接对训练好的模型进行可视化)

## 使用方法
### 数据准备
将下载的垃圾分类数据集解压到项目目录下，确保数据集结构如下：
garbage_classification/
├── battery/
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── ...其他类别...

### 训练模型
基本训练命令：
```bash
python main.py --data_dir ./garbage-dataset --model_type basic --epochs 200
```

使用数据增强：
```bash
python main.py --data_dir ./garbage-dataset --model_type resnet --epochs 200 --data_augmentation
```

使用改进的模型：
```bash
python main.py --data_dir ./garbage-dataset --model_type improved --epochs 200 --data_augmentation
```

### 参数说明
--data_dir: 数据集根目录，默认为当前项目目录下的garbage-dataset
--img_size: 输入图像尺寸，默认为224
--batch_size: 批次大小，默认为32
--data_augmentation: 是否使用数据增强
--model_type: 模型类型，默认为resnet，可选basic、resnet、improved
--epochs: 训练轮数，默认为200
--lr: 学习率，默认为0.001
--weight_decay: 权重衰减，默认为0.0001
--optimizer: 优化器类型，可选值为sgd或adam，默认为adam
--seed: 随机种子，默认为42
--save_dir: 模型保存目录，默认为当前目录下的checkpoints

### 输出文件
训练过程中和训练结束后，程序会生成以下文件：
- checkpoints/best_model.pth: 验证集上表现最好的模型
- training_history.png: 训练和验证损失及准确率曲线
- confusion_matrix.png: 混淆矩阵可视化
- class_metrics.png: 每个类别的精确率、召回率、F1-score
- model_predictions.png: 模型预测结果可视化
- gradcam_visualization.png: Grad-CAM可视化