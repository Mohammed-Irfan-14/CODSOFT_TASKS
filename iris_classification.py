import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("iris.csv")

print("First 5 Rows")
print(data.head())

print("\nDataset Information")
data.info()

print("\nDataset Shape")
print(data.shape)

print("\nMissing Values")
print(data.isnull().sum())

sns.pairplot(data, hue="species")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="species", data=data)
plt.title("Number of Flowers in Each Species")
plt.show()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel trained successfully")
print("Accuracy:", accuracy)