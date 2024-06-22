import numpy as np
import matplotlib.pyplot as plt
from util import *

np.random.seed(42)

def inverse_normal(A):
  return np.linalg.inv(A)

def inverse_gd(A, size, lr, num_iter, report_interval):
  '''
  A: numpy matrix of size (size, size)
  lr: numpy array of size (num_iter, )
  '''
  # initialize array
  X = np.random.randn(size, size)
  I = np.eye(size)
  A_t = np.transpose(A)
  A_inv_true = np.linalg.inv(A)

  # normalize X, still experimenting with different initialization
  norm_A = np.linalg.norm(A)
  norm_X = np.linalg.norm(X)
  X = X/(norm_A * norm_X)

  # training loop
  loss_rec = np.zeros((num_iter))
  loss_aux_rec = np.zeros((num_iter))
  for i in range(num_iter):
    res = A @ X # forward
    loss = np.sum(np.square(res - I)) # compute loss
    loss_aux = np.sum(np.square(X - A_inv_true))
    loss_rec[i] = loss
    loss_aux_rec[i] = loss_aux
    dX = 2 * A_t @ (res - I) # backward
    X -= lr[i] * dX # update 
    if i%report_interval == 0:
      print(f"Iteration {i}: loss = {loss}, loss w.r.t ground truth: {loss_aux}")
  return X, loss_rec, loss_aux_rec


def lr_schedule(num_iter, checkpoints):
  '''
  checkpoints: an array of tuples (index_prop, value)
  Must includes checkpoints for start and end of array
  '''
  lr = np.zeros((num_iter,))
  checkpoints.sort()
  for i in range(len(checkpoints)-1):
    index_cur = int(checkpoints[i][0] * (num_iter-1))
    index_next = int(checkpoints[i+1][0] * (num_iter-1))
    val_cur = checkpoints[i][1]
    val_next = checkpoints[i+1][1]
    lr[index_cur] = val_cur
    lr[index_next] = val_next
    lr[index_cur:index_next] = np.linspace(val_cur, val_next, index_next-index_cur, endpoint=False)
  return lr


def main():
  size = 5
  num_iter = int(100e3)
  lr = lr_schedule(num_iter, [(0, 0.05), (0.2, 0.025), (1, 0.01)])
  A = generate_non_singular_matrix_qr(size)
  A_inv_true = np.linalg.inv(A)
  A_inv2, loss_rec, loss_aux_rec = inverse_gd(A, size, lr, num_iter, report_interval=num_iter/100)
  # print(np.round(A_inv2@A, decimals=2))
  # print(np.round(A_inv_true@A, decimals=2))
  # diff = A_inv_true-A_inv2
  # print(np.round(diff, decimals=0))
  epoch_rec = [i for i in range(num_iter)]
  plot_loss(epoch_rec, loss_rec, loss_aux_rec, "prototype loss graph")


if __name__ == "__main__":
  main()
