import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import adaline as ad

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

print('From URL:', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
print(df.tail())

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', 0, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
ada1 = ad.AdalineGD(n_iter=15, eta=0.1).fit(X,y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

ada2 = ad.AdalineGD(n_iter=15, eta=0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.losses_) + 1), np.log10(ada2.losses_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()


# with standardization
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada_gd = ad.AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)

from plot_decision_regions import plot_decision_regions

plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_gd.losses_)+1), ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.tight_layout()
plt.show()


### Test Adaline SGD
ada_sgd = ad.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_sgd.losses_)+1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.tight_layout()
plt.show()

