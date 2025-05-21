import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
df = pd.read_csv("Crop_recommendation.csv")

# 清理无用的列
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 确保列名一致（统一大小写，避免因列名不同出错）
df.rename(columns={
    'phosphorus': 'Phosphorus',
    'potassium': 'Potassium'
}, inplace=True)

# 特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 分割数据，80%用于训练，20%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算模型准确度
accuracy = accuracy_score(y_test, y_pred)

# 输出准确度
print(f"模型在测试集上的准确率：{accuracy:.4f}")

# 输出更多的分类性能指标
print("\n详细的分类报告：")
print(classification_report(y_test, y_pred))

# 保存模型
joblib.dump(model, "crop_model.pkl")

# 输出训练完成信息
print("模型训练完成，并已保存为 crop_model.pkl")
