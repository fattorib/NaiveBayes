from get_mnist import fetch_mnist, fetch_fashionmnist
import numpy as np
import matplotlib.pyplot as plt

# ------- Data Preparation -------#

# X_train, Y_train, X_test, Y_test = fetch_mnist()

X_train, Y_train, X_test, Y_test = fetch_fashionmnist()

# Binarize - doesn't make sense to do NB this way if we do not
X_train = (X_train > 200).astype(np.int_)

X_test = (X_test > 200).astype(np.int_)


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, estimator, X, y, alpha=None, beta=None):

        if estimator == "MLE":
            self.mle_array = self.fit_MLE(X, y)

        if estimator == "MAP":
            assert (
                alpha is not None and beta is not None
            ), "Gamma priors for a,b must be defined when using MAP estimator"
            self.map_array = self.fit_MAP(X, y, alpha, beta)

    # ------- Fitting Params (MLE of Bernoulli param per feature, $\theta_{dc}$) -------#
    def fit_MLE(self, X, y):
        feature_dim = X.shape[1]
        self.num_classes = y.shape[1]

        mle_array = np.empty((feature_dim, self.num_classes))

        for i in range(self.num_classes):
            theta_MLE = np.mean(X_train[np.where(Y_train == i)], axis=0)
            mle_array[:, i] = theta_MLE
        return mle_array

    # ------- Fitting Params (MAP of Bernoulli param per feature, $\theta_{dc}$) -------#
    def fit_MAP(self, X, y, alpha, beta):
        feature_dim = X.shape[1]
        self.num_classes = y.shape[1]

        map_array = np.empty((feature_dim, self.num_classes))

        for i in range(self.num_classes):
            class_i = X_train[np.where(Y_train == i)]

            N_i = np.sum(class_i, axis=0)

            N_total = class_i.shape[0]

            theta_MAP = (alpha + N_i - 1) / (alpha + N_total + beta - 2)

            map_array[:, i] = theta_MAP

        return map_array

    def compute_conditional_class_dist(x, c, arr):
        # Explain this code
        class_dist = arr[:, c]

        prob = (class_dist ** (x)) * ((1 - class_dist) ** (1 - x))

        return np.prod(prob, axis=1)

    def predict(self, X, estimator):
        if estimator == "MLE":
            assert self.mle_array is not None

            preds_arr = np.empty(X.shape)

            for c in range(0, self.num_classes):

                out_c = self.compute_conditional_class_dist(X, c, arr=self.mle_array)
                preds_arr[:, c] = out_c

        if estimator == "MAP":
            assert self.map_array is not None

            preds_arr = np.empty(X.shape)
            for c in range(0, self.num_classes):
                out_c = self.compute_conditional_class_dist(X, c, arr=self.mle_array)
                preds_arr[:, c] = out_c

        return np.argmax(preds_arr, axis=1)

    def score(self, Y_pred, Y):
        print(f"Accuracy: {100*np.sum(Y_pred == Y)/Y_pred.shape[0]:.2f} %")


map_array = np.empty((784, 10))
mle_array = np.empty((784, 10))

alpha = 0.5
beta = 0.5

for i in range(0, 10):

    class_i = X_train[np.where(Y_train == i)]

    N_i = np.sum(class_i, axis=0)

    N_total = class_i.shape[0]

    theta_MAP = (alpha + N_i - 1) / (alpha + N_total + beta - 2)

    theta_MLE = np.mean(X_train[np.where(Y_train == i)], axis=0)

    map_array[:, i] = theta_MAP
    mle_array[:, i] = theta_MLE


def compute_conditional_class_dist(x, c, arr):

    class_dist = arr[:, c]

    prob = (class_dist ** (x)) * ((1 - class_dist) ** (1 - x))

    return np.prod(prob, axis=1)


preds_arr_MAP = np.empty((10000, 10))
preds_arr_MLE = np.empty((10000, 10))
for c in range(0, 10):

    out_c = compute_conditional_class_dist(x=X_test, c=c, arr=map_array)
    preds_arr_MAP[:, c] = out_c

    out_c = compute_conditional_class_dist(x=X_test, c=c, arr=mle_array)
    preds_arr_MLE[:, c] = out_c

preds = np.argmax(preds_arr_MAP, axis=1)
print(f"MAP Accuracy: {100*np.sum(preds == Y_test)/10000:.2f} %")

preds = np.argmax(preds_arr_MLE, axis=1)
print(f"MLE Accuracy: {100*np.sum(preds == Y_test)/10000:.2f} %")


# Compute diffs between params

# ------- Plotting -------#
fig, axs = plt.subplots(2, 5, figsize=(10, 6), constrained_layout=True)


i = 0
for col in range(0, 2):
    for ax in range(0, 5):
        # axs[col,ax].imshow((mle_array[:,i] - map_array[:,i]).reshape(28,28), cmap = 'RdBu')
        axs[col, ax].imshow((map_array[:, i]).reshape(28, 28))
        i += 1

plt.show()
