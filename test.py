# %%
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
# import sys
# sys.path.append("..") # 为了导⼊上层⽬录的d2lzh_pytorch
import d2lzh_pytorch as d2
# %%
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))
torch.gather(input=y_hat, dim=0, index=torch.tensor([[0],[1]]))
# %%
import tensorflow as tf
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]], dtype=tf.float32)
tf.gather(y_hat, [0, 1], axis=1, )
# tf.gather_nd(y_hat, [[0, 0], [1, 1]])
# %%