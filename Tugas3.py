import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# Drop missing values
df.dropna(inplace=True)

# Split features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
