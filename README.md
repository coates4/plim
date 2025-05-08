# PLIM: Projected Latent Influence for Model-Targeted Data Poisoning Attacks

This repository contains the official codebase for replicating results from the paper *"Exploring the Limits of Model-Targeted Indiscriminate Data Poisoning Attacks"*. It includes implementations of Gradient Canceling (GC), TGDA (Truncated Gradient Descent Ascent), and Gradient Matching (GradMatch) attacks across models like logistic regression (LR), multi-layer perceptrons (MLP), and convolutional neural networks (CNN).

---

## Environment Setup

I recommend using Python 3.8+ and running in VSCode, or alternatively you can use google colab.

# To reproduce results found in table 2 in paper
For all gradient canceling attacks, the code includes clipping after the gradient updates
if you wish to see the results without clipping, simply find it in code and remove it.
## Logistic Regression (MNIST)
python GC/mnist_lr-0.03.py

## Multi-layer Perceptron
python GC/mnist_nn-0.03-5e-1.py

## Convolutional Neural Network
python GC/mnist_cnn-0.03.py

## TGDA and Gradient Matching
python GradPC/TGDAGradMatch.py (--model lr, --model nn, --model cnn)

# Compare Results
If you wish to see if your results are similar to what I and the paper found simply check out the model image. The image is a table which has the results summarized.
