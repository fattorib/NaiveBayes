from numpy.random.mtrand import beta
from get_mnist import fetch_mnist, fetch_fashionmnist
import numpy as np
import matplotlib.pyplot as plt
from NaiveBayes import NaiveBayes

# ------- Data Preparation -------#

X_train, Y_train, X_test, Y_test = fetch_mnist()

# X_train, Y_train, X_test, Y_test = fetch_fashionmnist()


X_train = (X_train > 122).astype(np.int_)

X_test = (X_test > 122).astype(np.int_)


nb = NaiveBayes(num_classes=10)
MLE_array = nb.fit_MLE(X_train, Y_train)

preds = nb.predict(X_test, estimator="MLE")

score = nb.score(preds, Y_test)

print(f"Accuracy: {100*score:.2f} %")

nb.visualize_conditional_distributions(estimator="MLE")

MAP_array = nb.fit_MAP(X_train, Y_train, alpha=0.5, beta = 0.5)

preds = nb.predict(X_test, estimator="MAP")

score = nb.score(preds, Y_test)

print(f"Accuracy: {100*score:.2f} %")

nb.visualize_conditional_distributions(estimator="MAP")
