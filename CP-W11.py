import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# 下載資料集並讀取
df_train = pd.read_csv(r"Data/train.csv")
df_test = pd.read_csv(r"Data/test.csv")

# 印出前10筆資料
print(df_train.head(10))
print()

# 資料前處理
# 處理缺失值
df_train['Age'].fillna(df_train['Age'].median(), inplace=True) # 以中位數填補Age欄位
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True) # 田補Embarked欄位
df_train.drop(columns=['Cabin'], inplace=True) # 移除Cabin欄位(缺失值太多)

# 類別標籤轉換(轉成數值，方便模型訓練)
label_encoder = LabelEncoder()
df_train['Sex'] = label_encoder.fit_transform(df_train['Sex'])
df_train['Embarked'] = label_encoder.fit_transform(df_train['Embarked'])

# 特徵與標籤
X = df_train.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'])
y = df_train['Survived']

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練羅吉斯回歸模型
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 訓練決策樹模型
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

# 預測
y_pred_log_reg = log_reg.predict(X_test)
y_pred_tree_clf = tree_clf.predict(X_test)

# 評估模型
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# 羅吉斯回歸評估
log_reg_metrics = evaluate_model(y_test, y_pred_log_reg)
print(f"Logistic Regression - Accuracy: {log_reg_metrics[0]:.2f}, Precision: {log_reg_metrics[1]:.2f}, Recall: {log_reg_metrics[2]:.2f}, F1-score: {log_reg_metrics[3]:.2f}")

# 決策樹評估
tree_clf_metrics = evaluate_model(y_test, y_pred_tree_clf)
print(f"Decision Tree - Accuracy: {tree_clf_metrics[0]:.2f}, Precision: {tree_clf_metrics[1]:.2f}, Recall: {tree_clf_metrics[2]:.2f}, F1-score: {tree_clf_metrics[3]:.2f}")

# 混淆矩陣圖
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
conf_matrix_tree_clf = confusion_matrix(y_test, y_pred_tree_clf)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(conf_matrix_log_reg).plot(ax=ax[0])
ax[0].set_title('Logistic Regression Confusion Matrix')
ConfusionMatrixDisplay(conf_matrix_tree_clf).plot(ax=ax[1])
ax[1].set_title('Decision Tree Confusion Matrix')
plt.show()