import pandas as pd
from io import StringIO

csv_data = \
    """
    A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,
    """

df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull().sum())

# Access NumPy array
print(df.values)

# remove rows with missing values
print(df.dropna(axis=0))

# drop columns with missing values
print(df.dropna(axis=1))

# only drop rows where all columns are NaN
df.dropna(how='all')

# drop rows that have fewer than 4 real values
df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

# impute missing values
print("\nImputed data....\n")
from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

# this would do the same thing
df.fillna(df.mean())


# Handling categorical data
print("categorical data")
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

# map ordinal feature to integer
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print(df)

# if we want to reverse the mapping
inv_size_mapping = {v:k for k, v in size_mapping.items()}
#df['size'] = df['size'].map(inv_size_mapping)
#print(df)

# it's best practice to convert class labels to integers.
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
inv_class_mapping = {v:k for k,v in class_mapping.items()}

# alternatively use LabelEncoder class in skl
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
# inverse
class_le.inverse_transform(y)

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:, 0])
print(X)

# use one-hot encoding for colors
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
# selectively transform columns in a multi-feature array
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1,2])
])
print(c_transf.fit_transform(X).astype(float))

# even more convenient - create dummy features via one-hot encoding
print(pd.get_dummies(df[['price', 'color', 'size']]))
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))

# drop a redundant column via OHE
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
    ('onehot', color_ohe, [0]),
    ('nothing', 'passthrough', [1,2])
])
print(c_transf.fit_transform(X).astype(float))


