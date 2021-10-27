# -*- coding: utf-8 -*-
#
# Generative Adversarial Network on toy examples
#
# partially inspired from:
#   - https://github.com/devnag/pytorch-generative-adversarial-networks
#   - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#   - https://github.com/AntoinePlumerault/AVAE/  (toy example)
import random
import numpy as np
import matplotlib.pyplot as plt

# torch stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Abseil utils from Google https://github.com/abseil/abseil-py
from absl import app, flags

# we use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# return N data drawn according to the wanted density
def f_data(N, model='circle'):
  eps = np.random.randn(N) # Gaussian noise
  if model == 'circle':
    t = np.random.rand(N) # Uniform
    return np.column_stack((TODO,TODO))

  z1 = np.random.randn(N) # Gaussian
  if model == 'simple_sin':
    return np.column_stack((TODO,TODO))
  elif model == 'double_sin':
    z2 = np.random.randn(N) # Gaussian (2)
    return np.column_stack((3*z1+0.1*eps,np.cos(3*z1)+np.tanh(3*z2)+0.1*eps))

class Generator(nn.Module):
  def __init__(self, sz_latent,sz_hidden):
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(TODO,TODO)
    self.fc2 = nn.Linear(TODO,TODO)
    self.fout = nn.Linear(TODO,TODO)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fout(x)
    return x

class Discriminator(nn.Module):
  def __init__(self, sz):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(TODO,sz)
    self.fc2 = nn.Linear(sz,TODO)
    self.fc3 = nn.Linear(int(sz/2),TODO)
    self.fout = nn.Linear(int(sz/4),TODO)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = TODO # decision (proba)
    return x

def extract(v):
    return v.data.storage().tolist()

def main(argv):
  latent_dim = 2 # 1 for 1D manifold
  G = Generator(latent_dim,32).to(device)
  D = Discriminator(32).to(device)

  criterion = nn.BCELoss()
  d_optimizer = optim.SGD(TODO, lr=1e-3, momentum=0.8)
  g_optimizer = optim.SGD(TODO, lr=1e-3, momentum=0.8)
  
  batch_size = 32

  for epoch in range(2000):
    for ii in range(20):  # train D for 20 steps
      D.zero_grad() # could be d_optimizer.zero_grad() since the optimizer is specific to the model

      # train D on real data
      d_real_data = (torch.FloatTensor(f_data(batch_size,FLAGS.model))).to(device)
      d_real_decision = D(d_real_data)
      d_real_error = criterion(d_real_decision, torch.TODO([batch_size,1])).to(device)
      d_real_error.backward() # compute/store gradients, but don't change params

      # train D on fake data
      d_gen_seed = (torch.FloatTensor( torch.randn(batch_size,latent_dim ) )).to(device)  # TODO rand ou randn ?
      d_fake_data = G( d_gen_seed ).detach()  # detach to avoid training G on these labels
      d_fake_decision = D(d_fake_data)
      d_fake_error = criterion(d_fake_decision, torch.TODO([batch_size,1]).to(device))
      d_fake_error.backward()
      d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

      dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

    for ii in range(20):  # train G for 20 steps
      G.zero_grad()

      g_gen_seed = (torch.FloatTensor( torch.randn(batch_size,latent_dim ))).to(device)
      g_fake_data = G( g_gen_seed )
      dg_fake_decision = D(g_fake_data)
      g_error = criterion(dg_fake_decision, torch.TODO([batch_size,1]).to(device))  # Train G to pretend it's genuine

      g_error.backward()
      g_optimizer.step()  # Only optimizes G's parameters

      ge = extract(g_error)[0]
    if epoch % 20 ==0:
      print("Epoch %s: D (%1.4f real_err, %1.4f fake_err) G (%1.4f err) " % (epoch, dre, dfe, ge))

    if epoch % 60 == 0:
      g_gen_seed = (torch.FloatTensor( torch.randn(1000,latent_dim ))).to(device)
      g_fake_data = G( g_gen_seed ).detach().to("cpu")
      plt.cla()

      # plot ground truth
      if FLAGS.model == "circle":
        t=np.arange(0,1.1,0.025)
        plt.plot(3*np.cos(t*2*np.pi),3*np.sin(t*2*np.pi), 'r-')
      if FLAGS.model == "simple_sin":
        xx = np.arange(-3,3,0.25)
        plt.plot(3*xx,np.cos(3*xx), 'r-')
      if FLAGS.model == "double_sin":
        xx = np.arange(-3,3,0.25)
        plt.plot(3*xx,np.cos(3*xx)+1, 'r-')
        plt.plot(3*xx,np.cos(3*xx)-1, 'r-')

      plt.plot(g_fake_data[:,0],g_fake_data[:,1],'b.')
      plt.draw()
      plt.pause(0.001)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_enum('model', 'circle', ['circle', 'simple_sin', 'double_sin'], "")
    app.run(main)
