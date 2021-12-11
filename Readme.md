
# Naive Bayes - MNIST

Short example fitting a Naive Bayes classifier to binarized MNIST data with visalization of class conditional densities. In this example, every pixel value is either a 0 or 1. For each class from 0 to 9, we are then estimating $\theta_{dc}$ for a Bernoulli distribution where d is the pixel number and c is the class. 

## Bernoulli class conditional densities - MLE

Test set accuracy: 84.39%. 

## Bernoulli class conditional densities - MAP

Test set accuracy: 84.44%. Fit with a Beta(0.5,0.5) prior.


