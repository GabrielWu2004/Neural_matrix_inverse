# Neural Matrix Inverse

This is a fun weekend project where I try to find the inverse of a matrix through gradient descent.

## Idea
The process of inverting a matrix can be thought of as the following: 
Given a (square) matrix $A$, find another matrix $A^{-1}$ such as $A^{-1}A=I$. Therefore, we can treat this as an optimization problem ${A^{-1}=\text{argmin}_B(AB-I)}$. That is, we want to find a matrix $B$ such that the product of $AB$ is as close of the identity matrix as possible. In another word, we can create a neural network with only one linear layer, such that, after trained on a single input matrix for multiple iterations, the weight of the linear network is the inverse of the input matrix. Note that in this case the neural network is not supposed to perform "inference" of any kind. Instead, the network itself (i.e. its weight) is our desired matrix inverse.

Let $X$ be the input of the neural network and let $W$ be the weight matrix of the linear layer. Both have shape (size, size). The network performs simple matrix multiplication $WX$. We can define a loss function as the mean-squared error $L = \frac{1}{N}||WX-I||_2^2$ and minimize it using gradient descent.

## Implementation
I first implemented a small prototype in numpy to invert a 5x5 matrix. After seeing that the loss are indeed going down after a few thousand iterations, I scaled things up in Pytorch to invert a 16x16 matrix. In addition to optimizing and reporting the mean-squared loss between $WX$ and $I$, I also periodically report the auxilibary loss between $W$ and the ground truth $X^{-1}$ calculated using the traiditional method with numpy. This loss is not used for optimization but for illustration. I use ADAM optimizer and a custom learning rate scheduler, which starts the learning rate at 0.05 and reduces it to 0.025 and 0.01 after 10k and 100k iterations, respectively. 

## Result
Depending on the initialization, the neural network is able to approximate the matrix inverse to high accuracy (loss in the order of 1e-8). However, there are many caveats.Here are some observations during the optimization process:
1. It's significantly easier to optimize smaller matrice than larger ones. Seems like the high dimensionality creates some serious trouble for the gradient descent optimization.
2. Training is often stuck at local minimum. This is very weird because the loss function seems to be convex and there is only one linear layer. 
3. Training results heavily depend on initialization. Sometimes the auxiliary errors stays in the thousands after 100k iterations; other times it drops to <1 after 10k. 

Some training runs are summarized in the loss graphs. 