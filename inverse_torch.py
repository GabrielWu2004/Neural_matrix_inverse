import numpy as np
import matplotlib.pyplot as plt
from torch.nn import modules

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util import *

class inverseNN(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.inverse = nn.Linear(size, size, bias=False) # x@A.T
  
  def forward(self, x):
    return self.inverse(x)


def fitting(size, input, lr, num_epochs):
  '''
  size (int): the size of the input matrix
  input (np.ndarray): numpy matrix to be inverted of shape (size, size)
  '''

  # initialize fitter
  fitter = inverseNN(size).cuda()
  input = torch.tensor(input, dtype=torch.float32).cuda()
  target = torch.tensor(np.eye(size), dtype=torch.float32).cuda()
  criterion = nn.MSELoss()
  optimizer = optim.Adam(fitter.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-5)

  # ground truth inverse
  inverse_gt = torch.linalg.inv(input)
  inverse_gt_t = torch.transpose(inverse_gt, 1, 0)

  # logging
  epoch_log = np.zeros(int(num_epochs//1000))
  loss_log = np.zeros(int(num_epochs//1000))
  aux_loss_log = np.zeros(int(num_epochs//1000))

  # fitting
  for epoch in range (num_epochs):
    optimizer.zero_grad()
    output = fitter(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # log result
    if (epoch+1) % (100) == 0:
      inverse_fit_t = fitter.inverse.weight.data
      aux_loss = criterion(inverse_fit_t, inverse_gt_t)
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}, Aux loss: {aux_loss.item():.8f}')
      print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
      index = (epoch+1)//1000 - 1
      epoch_log[index] = epoch+1
      loss_log[index] = loss.item()
      aux_loss_log[index] = aux_loss.item()
    
  inverse_fit = torch.transpose(fitter.inverse.weight.data.cpu(), 0, 1).numpy()
  return inverse_fit, epoch_log, loss_log, aux_loss_log

if __name__ == "__main__":
  size = 10000
  num_epochs = int(1e5)
  lr = 0.001
  matrix = generate_non_singular_matrix_qr(size)
  inverse_fit, epoch_log, loss_log, aux_loss_log = fitting(size, matrix, lr, num_epochs)
  plot_loss(epoch_log, loss_log, aux_loss_log, "torch training loss")