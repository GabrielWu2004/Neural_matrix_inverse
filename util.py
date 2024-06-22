import numpy as np
import matplotlib.pyplot as plt

def generate_non_singular_matrix_qr(n):
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    R = np.triu(np.random.rand(n, n))
    return np.dot(Q, R)


def plot_loss(epoch_log, loss_log, aux_loss_log, name):
  plt.loglog(epoch_log, loss_log, label='loss')
  plt.loglog(epoch_log, aux_loss_log, label='loss w.r.t. ground truth')
  plt.title('Training loss')
  plt.xlabel('num_iter')
  plt.ylabel('loss')
  plt.legend()
  plt.savefig(f"{name}.png")
  plt.show()