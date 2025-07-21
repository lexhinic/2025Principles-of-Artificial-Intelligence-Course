# 人工智能基础 - 第二次编程作业：信用数据分析问题
# German Credit Data Analysis using PyTorch

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ucimlrepo import fetch_ucirepo

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 加载德国信用数据集
# 使用ucimlrepo库加载数据
german_credit = fetch_ucirepo(id=144)

# 提取特征和目标变量
X = german_credit.data.features
y = german_credit.data.targets

# 查看数据集基本信息
print("数据集形状:", X.shape)
print("目标变量分布:")
print(y.value_counts())

# 2. 数据预处理
# 检查缺失值
print("\n缺失值统计:")
print(X.isnull().sum())

# 检查数据类型
print("\n数据类型:")
print(X.dtypes)

# 确定数值型和类别型特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\n数值型特征:", numeric_features)
print("类别型特征:", categorical_features)

# 创建特征处理的转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. 数据集划分
# 划分训练集、验证集和测试集（60%/20%/20%）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("\n数据集划分:")
print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# 4. Logistic回归模型 (使用sklearn)
# 创建Logistic回归的Pipeline
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 训练Logistic回归模型
lr_pipeline.fit(X_train, y_train)

# 在测试集上评估模型
y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("\nLogistic回归模型性能:")
print(f"准确率: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_lr))

# 计算ROC曲线和AUC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# 5. 深度神经网络模型 (使用PyTorch)
# 创建预处理好的数据
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# 将数据转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed)
y_train_tensor = torch.FloatTensor(y_train.values)

X_val_tensor = torch.FloatTensor(X_val_processed.toarray() if hasattr(X_val_processed, "toarray") else X_val_processed)
y_val_tensor = torch.FloatTensor(y_val.values)

X_test_tensor = torch.FloatTensor(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed)
y_test_tensor = torch.FloatTensor(y_test.values)

# 创建PyTorch数据集类
class CreditDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 创建数据加载器
train_dataset = CreditDataset(X_train_tensor, y_train_tensor)
val_dataset = CreditDataset(X_val_tensor, y_val_tensor)
test_dataset = CreditDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义神经网络模型
class CreditNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(CreditNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience=10):
    model.to(device)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
            # 计算准确率
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 保存训练历史
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        
        # 输出当前epoch的训练情况
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # 保存最佳模型
            best_model_weights = model.state_dict().copy()
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_weights)  # 恢复最佳模型权重
                break
    
    # 如果没有早停，也要恢复到最佳模型
    if epoch == epochs - 1:
        model.load_state_dict(best_model_weights)
    
    return model, training_history

# 评估函数
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_probs), np.array(all_labels)

# 超参数实验
# 实验1: 不同隐藏层节点数
hidden_units_list = [32, 64, 128]
results_hidden = []

input_dim = X_train_tensor.shape[1]

for hidden_dim in hidden_units_list:
    model = CreditNet(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=0.2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n训练隐藏层节点数为 {hidden_dim} 的模型")
    model, _ = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10)
    
    val_acc, _, _ = evaluate_model(model, val_loader)
    results_hidden.append({'hidden_dim': hidden_dim, 'val_acc': val_acc})
    
    print(f"隐藏层节点数: {hidden_dim}, 验证集准确率: {val_acc:.4f}")

# 实验2: 不同Dropout率
dropout_rates = [0.1, 0.2, 0.3, 0.5]
results_dropout = []

best_hidden_dim = max(results_hidden, key=lambda x: x['val_acc'])['hidden_dim']

for dropout_rate in dropout_rates:
    model = CreditNet(input_dim=input_dim, hidden_dim=best_hidden_dim, dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n训练Dropout率为 {dropout_rate} 的模型")
    model, _ = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10)
    
    val_acc, _, _ = evaluate_model(model, val_loader)
    results_dropout.append({'dropout_rate': dropout_rate, 'val_acc': val_acc})
    
    print(f"Dropout率: {dropout_rate}, 验证集准确率: {val_acc:.4f}")

# 选择最佳超参数训练最终模型
best_hidden_dim = max(results_hidden, key=lambda x: x['val_acc'])['hidden_dim']
best_dropout_rate = max(results_dropout, key=lambda x: x['val_acc'])['dropout_rate']

print(f"\n最佳超参数:")
print(f"隐藏层节点数: {best_hidden_dim}")
print(f"Dropout率: {best_dropout_rate}")

# 使用最佳超参数训练最终模型
final_model = CreditNet(input_dim=input_dim, hidden_dim=best_hidden_dim, dropout_rate=best_dropout_rate)
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001)

print("\n训练最终模型")
final_model, history = train_model(final_model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10)

# 在测试集上评估最终DNN模型
test_acc, test_probs_dnn, test_labels = evaluate_model(final_model, test_loader)
y_pred_dnn = (test_probs_dnn > 0.5).astype(int)

print("\n深度神经网络模型性能:")
print(f"准确率: {test_acc:.4f}")
print("\n分类报告:")
print(classification_report(test_labels, y_pred_dnn))

# 计算深度神经网络的ROC曲线和AUC
fpr_dnn, tpr_dnn, _ = roc_curve(test_labels, test_probs_dnn)
roc_auc_dnn = auc(fpr_dnn, tpr_dnn)

# 绘制训练历史
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 6. 绘制ROC曲线比较两种模型
plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
plt.plot(fpr_dnn, tpr_dnn, label=f'Deep Neural Network (AUC = {roc_auc_dnn:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 7. 比较模型在各评估指标上的表现
def calculate_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auroc = auc(fpr, tpr)
    
    return {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'auROC': auroc
    }

# 计算两个模型的评估指标
lr_metrics = calculate_metrics(y_test, y_pred_lr, y_proba_lr)
dnn_metrics = calculate_metrics(test_labels, y_pred_dnn, test_probs_dnn)

# 创建对比表格
metrics_df = pd.DataFrame({
    'Metric': list(lr_metrics.keys()),
    'Logistic Regression': list(lr_metrics.values()),
    'Deep Neural Network': list(dnn_metrics.values())
})

print("\n两种模型评估指标对比:")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 绘制指标对比柱状图
plt.figure(figsize=(12, 8))
metrics_df.set_index('Metric').plot(kind='bar')
plt.title('Performance Metrics Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 8. (选做) 年龄特征上的偏见分析
# 提取原始年龄数据
age_column = [col for col in X.columns if 'age' in col.lower()][0]
age_data = X[age_column].copy()

# 将年龄分成几个区间
age_bins = [0, 30, 40, 50, 60, 100]
age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
age_groups = pd.cut(age_data, bins=age_bins, labels=age_labels, right=False)

# 按照年龄组统计实际标签分布
age_actual = pd.DataFrame({
    'Age Group': age_groups,
    'Credit Status': y.values.flatten()
}).groupby('Age Group')['Credit Status'].value_counts(normalize=True).unstack()

# 按照年龄组统计Logistic回归模型的预测分布
X_with_age = X.copy()
X_with_age['Age Group'] = age_groups
lr_preds = lr_pipeline.predict(X)
age_lr_pred = pd.DataFrame({
    'Age Group': X_with_age['Age Group'],
    'Predicted': lr_preds
}).groupby('Age Group')['Predicted'].value_counts(normalize=True).unstack()

# 按照年龄组统计深度神经网络模型的预测分布
final_model.eval()
with torch.no_grad():
    X_processed = preprocessor.transform(X)
    X_tensor = torch.FloatTensor(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed).to(device)
    dnn_probs = final_model(X_tensor).squeeze().cpu().numpy()
    dnn_preds = (dnn_probs > 0.5).astype(int)

age_dnn_pred = pd.DataFrame({
    'Age Group': X_with_age['Age Group'],
    'Predicted': dnn_preds
}).groupby('Age Group')['Predicted'].value_counts(normalize=True).unstack()

# 绘制实际分布与预测分布的对比
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 实际分布
age_actual.plot(kind='bar', ax=axes[0], title='Actual Credit Status by Age')
axes[0].set_ylabel('Proportion')
axes[0].set_ylim(0, 1)
axes[0].legend(['Bad', 'Good'])

# Logistic回归预测分布  
age_lr_pred.plot(kind='bar', ax=axes[1], title='Logistic Regression Predictions by Age')
axes[1].set_ylabel('Proportion')
axes[1].set_ylim(0, 1)
axes[1].legend(['Bad', 'Good'])

# 深度神经网络预测分布
age_dnn_pred.plot(kind='bar', ax=axes[2], title='DNN Predictions by Age')
axes[2].set_ylabel('Proportion')
axes[2].set_ylim(0, 1)
axes[2].legend(['Bad', 'Good'])

plt.tight_layout()
plt.show()

# 计算每个年龄组中两种模型的准确率
age_acc = {}

for age_group in age_labels:
    # 获取该年龄组的索引
    age_idx = X_with_age[X_with_age['Age Group'] == age_group].index
    
    # 计算两种模型在该年龄组上的准确率
    lr_acc = accuracy_score(y.iloc[age_idx], lr_preds[age_idx])
    dnn_acc = accuracy_score(y.iloc[age_idx], dnn_preds[age_idx])
    
    age_acc[age_group] = {'Logistic Regression': lr_acc, 'Deep Neural Network': dnn_acc}

# 创建准确率对比表格
age_acc_df = pd.DataFrame.from_dict(age_acc, orient='index')
print("\n各年龄组模型准确率:")
print(age_acc_df)

# 绘制各年龄组准确率对比
plt.figure(figsize=(10, 6))
age_acc_df.plot(kind='bar')
plt.title('Model Accuracy by Age Group')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 计算两种模型在各年龄组的偏差度
# 偏差度定义为：|预测的Good比例 - 实际的Good比例|
bias_df = pd.DataFrame(index=age_labels, columns=['LR Bias', 'DNN Bias'])

for age_group in age_labels:
    if 1 in age_actual.loc[age_group] and 1 in age_lr_pred.loc[age_group] and 1 in age_dnn_pred.loc[age_group]:
        actual_good_rate = age_actual.loc[age_group, 1]
        lr_good_rate = age_lr_pred.loc[age_group, 1]
        dnn_good_rate = age_dnn_pred.loc[age_group, 1]
        
        bias_df.loc[age_group, 'LR Bias'] = abs(lr_good_rate - actual_good_rate)
        bias_df.loc[age_group, 'DNN Bias'] = abs(dnn_good_rate - actual_good_rate)

print("\n各年龄组模型偏差度:")
print(bias_df)

# 绘制偏差度对比
plt.figure(figsize=(10, 6))
bias_df.plot(kind='bar')
plt.title('Model Bias by Age Group')
plt.ylabel('Bias (absolute difference in Good prediction rate)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 9. (作业第1问) Logistic回归的随机梯度下降推导
# 注：这部分仅提供PyTorch实现的SGD Logistic回归，以展示实际计算过程
# 完整的数学推导应在报告中展开

class LogisticRegressionSGD(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionSGD, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 创建自定义的SGD Logistic回归模型
sgd_model = LogisticRegressionSGD(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(sgd_model.parameters(), lr=0.01)

# 训练SGD Logistic回归模型
print("\n训练SGD Logistic回归模型")
epochs = 100
losses = []

for epoch in range(epochs):
    epoch_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        # 前向传播
        outputs = sgd_model(features).squeeze()
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * features.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    losses.append(epoch_loss)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}')

# 在测试集上评估SGD Logistic回归模型
sgd_acc, sgd_probs, _ = evaluate_model(sgd_model, test_loader)
print(f"\nSGD Logistic回归模型准确率: {sgd_acc:.4f}")

# 绘制损失下降曲线
plt.figure(figsize=(8, 6))
plt.plot(losses)
plt.title('SGD Logistic Regression Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()