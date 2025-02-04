import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

file_path = "IRIS.csv"
df = pd.read_csv(file_path)

print("\nHead of the dataset:")
print(df.head())

print("\nTail of the dataset:")
print(df.tail())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

X = df.drop("species", axis=1)
y = df["species"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFirst 5 scaled feature values:")
print(X_scaled[:5])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel accuracy:", accuracy)

new_data = [[5.7, 3.0, 4.2, 1.2]]
new_data_scaled = scaler.transform(new_data)
new_prediction = model.predict(new_data_scaled)
print("Prediction for the new flower:", new_prediction[0])

new_data = [[5.7, 3.0, 4.2, 4.5], [6.0, 2.8, 5.1, 2.4]]
new_data_scaled = scaler.transform(new_data)
new_predictions = model.predict(new_data_scaled)
print("Predictions for the new flowers:", new_predictions)

sns.set_style("whitegrid")

sns.pairplot(df, hue="species", markers=["o", "s", "D"], palette="Set2")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="species", y="sepal_length", palette="Set1")
plt.title("Boxplot of Sepal Length by Species")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="species", y="petal_length", palette="coolwarm")
plt.title("Violin Plot of Petal Length by Species")
plt.show()

df.hist(figsize=(10, 6), color="skyblue", edgecolor="black")
plt.suptitle("Histogram of Iris Features", y=1.02)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
