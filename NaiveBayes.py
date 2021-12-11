import numpy as np
import matplotlib.pyplot as plt


class NaiveBayes:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, estimator, X, y, alpha=None, beta=None):

        if estimator == "MLE":
            self.mle_array = self.fit_MLE(X, y)

        if estimator == "MAP":
            assert (
                alpha is not None and beta is not None
            ), "Beta priors for a,b must be defined when using MAP estimator"
            self.map_array = self.fit_MAP(X, y, alpha, beta)

    # ------- Fitting Params (MLE of Bernoulli param per feature, $\theta_{dc}$) -------#
    def fit_MLE(self, X, y):
        feature_dim = X.shape[1]

        mle_array = np.empty((feature_dim, self.num_classes))

        for i in range(self.num_classes):
            theta_MLE = np.mean(X[np.where(y == i)], axis=0)
            mle_array[:, i] = theta_MLE
        self.mle_array = mle_array

    # ------- Fitting Params (MAP of Bernoulli param per feature, $\theta_{dc}$) -------#
    def fit_MAP(self, X, y, alpha, beta):
        feature_dim = X.shape[1]

        map_array = np.empty((feature_dim, self.num_classes))

        for i in range(self.num_classes):
            class_i = X[np.where(y == i)]

            N_i = np.sum(class_i, axis=0)

            N_total = class_i.shape[0]

            theta_MAP = (alpha + N_i - 1) / (alpha + N_total + beta - 2)

            map_array[:, i] = theta_MAP

        self.map_array = map_array

    def compute_conditional_class_dist(self, x, c, arr):
        
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
                out_c = self.compute_conditional_class_dist(X, c, arr=self.map_array)
                preds_arr[:, c] = out_c

        return np.argmax(preds_arr, axis=1)

    def score(self, Y_pred, Y):
        return np.sum(Y_pred == Y) / Y_pred.shape[0]

    def visualize_conditional_distributions(self, estimator):

        if estimator == "MLE":
            assert self.mle_array is not None, "Requires fitting MLE first"
            conditional_array = self.mle_array

        if estimator == "MAP":
            assert self.map_array is not None, "Requires fitting MAP first"

            conditional_array = self.map_array

        ncols = 5
        nrows = 2

        # create the plots
        fig = plt.figure()
        axes = [
            fig.add_subplot(nrows, ncols, r * ncols + c + 1)
            for r in range(0, nrows)
            for c in range(0, ncols)
        ]

        i = 0
        for ax in axes:
            ax.imshow((conditional_array[:, i]).reshape(28, 28))
            i += 1

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()
