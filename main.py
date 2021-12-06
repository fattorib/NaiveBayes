from get_mnist import fetch_mnist, fetch_fashionmnist
import numpy as np
import matplotlib.pyplot as plt

#------- Data Preparation -------#

X_train, Y_train, X_test, Y_test = fetch_mnist()

#Binarize - doesn't make sense to do NB this way if we do not
X_train = (X_train > 200).astype(np.int_)

X_test = (X_test > 200).astype(np.int_)


#------- Fitting Params (MLE of Bernoulli param per feature, $\theta_{dc}$) -------#

#------- Fitting Params (MAP of Bernoulli param per feature, $\theta_{dc}$) -------#
map_array = np.empty((784,10))
mle_array = np.empty((784,10))

alpha = 0.5
beta = 0.5

for i in range(0,10):
    
    class_i = X_train[np.where(Y_train == i)]

    N_i = np.sum(class_i, axis = 0)

    N_total = class_i.shape[0]

    theta_MAP = (alpha + N_i - 1)/(alpha + N_total + beta - 2)
    
    theta_MLE = np.mean(X_train[np.where(Y_train == i)], axis = 0)

    map_array[:,i] = theta_MAP
    mle_array[:,i] = theta_MLE


def compute_conditional_class_dist(x, c, arr):

    class_dist = arr[:,c]

    prob = (class_dist**(x))*((1-class_dist)**(1-x))

    return np.prod(prob, axis = 1)


preds_arr_MAP = np.empty((10000,10))
preds_arr_MLE = np.empty((10000,10))
for c in range(0,10):   

    out_c = compute_conditional_class_dist(x = X_test, c = c, arr = map_array)
    preds_arr_MAP[:,c] = out_c

    out_c = compute_conditional_class_dist(x = X_test, c = c, arr = mle_array)
    preds_arr_MLE[:,c] = out_c

preds = np.argmax(preds_arr_MAP, axis = 1)
print(f'MAP Accuracy: {100*np.sum(preds == Y_test)/10000:.2f} %')

preds = np.argmax(preds_arr_MLE, axis = 1)
print(f'MLE Accuracy: {100*np.sum(preds == Y_test)/10000:.2f} %')


#------- Plotting -------#
fig, axs = plt.subplots(2, 5, figsize=(10, 6), constrained_layout=True)

i = 0
for col in range(0,2):
    for ax in range(0,5):
        axs[col,ax].imshow(mle_array[:,i].reshape(28,28))
        i += 1
plt.show()


