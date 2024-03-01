# -*- coding: utf-8 -*-
#
# Generative Adversarial Network on toy examples in 3D
#
# partially inspired from:
#   - https://github.com/devnag/pytorch-generative-adversarial-networks
#   - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#   - https://github.com/AntoinePlumerault/AVAE/  (toy example)
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # dessin cube (ground truth)

# avoid a numpy warning (FIXME proprement...)
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

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
def f_data(N, model='helix'):
  eps = np.random.randn(N) # Gaussian noise
  if model == 'helix':
    t = np.random.rand(N)*2-1 # Uniform
    return np.column_stack((4*np.cos(t*2*np.pi)+0.1*eps,4*np.sin(t*2*np.pi)+0.1*eps,4*t+0.1*eps))
  if model == 'bike_accident':
    N=int(N/2)
    eps = np.random.randn(N)
    t = np.random.rand(N)*4-2
    r = 4
    x1 = r * np.sin(t*2*np.pi)+0.1*eps
    y1 = r * np.cos(t*2*np.pi)+0.1*eps
    z1 = t*0
    d1=np.column_stack((x1,y1,z1))

    x2 = r + r * np.sin(t*2*np.pi)+0.1*eps
    y2 = t*0
    z2 = r * np.cos(t*2*np.pi)+0.1*eps
    d2=np.column_stack((x2,y2,z2))
    return np.concatenate((d1,d2), axis=0)
  if model == 'saddle_point':
    x = np.random.rand(N)*4-2
    y = np.random.rand(N)*4-2
    z = x**2-y**2+0.1*eps
    return np.column_stack((x,y,z))
  if model == 'full_cube':
    x = np.random.rand(N)*4-2
    y = np.random.rand(N)*4-2
    z = np.random.rand(N)*4-2
    return np.column_stack((x,y,z))

def get_face_cube():
  faces = []
  faces.append(np.zeros([5,3]))
  faces.append(np.zeros([5,3]))
  faces.append(np.zeros([5,3]))
  faces.append(np.zeros([5,3]))
  faces.append(np.zeros([5,3]))
  faces.append(np.zeros([5,3]))
  # Bottom face
  faces[0][:,0] = [-2,-2,2,2,-2]
  faces[0][:,1] = [-2,2,2,-2,-2]
  faces[0][:,2] = [-2,-2,-2,-2,-2]
  # Top face
  faces[1][:,0] = [-2,-2,2,2,-2]
  faces[1][:,1] = [-2,2,2,-2,-2]
  faces[1][:,2] = [2,2,2,2,2]
  # Left Face
  faces[2][:,0] = [-2,-2,-2,-2,-2]
  faces[2][:,1] = [-2,2,2,-2,-2]
  faces[2][:,2] = [-2,-2,2,2,-2]
  # Left Face
  faces[3][:,0] = [2,2,2,2,2]
  faces[3][:,1] = [-2,2,2,-2,-2]
  faces[3][:,2] = [-2,-2,2,2,-2]
  # front face
  faces[4][:,0] = [-2,2,2,-2,-2]
  faces[4][:,1] = [-2,-2,-2,-2,-2]
  faces[4][:,2] = [-2,-2,2,2,-2]
  # front face
  faces[5][:,0] = [-2,2,2,-2,-2]
  faces[5][:,1] = [2,2,2,2,2]
  faces[5][:,2] = [-2,-2,2,2,-2]
  return faces

class Generator(nn.Module):
  def __init__(self, sz_latent,sz_hidden):
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(sz_latent,sz_hidden)
    self.fc2 = nn.Linear(sz_hidden,sz_hidden)
    self.fout = nn.Linear(sz_hidden,3)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fout(x)
    return x

class Discriminator(nn.Module):
  def __init__(self, sz):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(3,sz)
    self.fc2 = nn.Linear(sz,int(sz/2))
    self.fc3 = nn.Linear(int(sz/2),int(sz/4))
    self.fout = nn.Linear(int(sz/4),1) # output size==1 : raw score
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = torch.sigmoid(self.fout(x)) # sigmoid(raw score) -> decision (proba)
    return x

def extract(v):
    return v.data.storage().tolist()

def main(argv):
  latent_dim = FLAGS.latent_dim # default 2; 1 for 1D manifold
  G = Generator(latent_dim,32).to(device)
  D = Discriminator(32).to(device)

  criterion = nn.BCELoss()
  d_optimizer = optim.SGD(D.parameters(), lr=1e-3, momentum=0.8)
  g_optimizer = optim.SGD(G.parameters(), lr=1e-3, momentum=0.8)
  
  batch_size = 32

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  face_cube = get_face_cube()
  for epoch in range(FLAGS.epochs):
    for ii in range(20):  # train D for 20 steps
      D.zero_grad() # could be d_optimizer.zero_grad() since the optimizer is specific to the model

      # train D on real data
      d_real_data = (torch.FloatTensor(f_data(batch_size,FLAGS.model))).to(device)
      d_real_decision = D(d_real_data)
      d_real_error = criterion(d_real_decision, (torch.ones([batch_size,1])).to(device))  # ones = true
      d_real_error.backward() # compute/store gradients, but don't change params

      # train D on fake data
      d_gen_seed = (torch.FloatTensor( torch.randn(batch_size,latent_dim ))).to(device)  # TODO rand ou randn ?
      d_fake_data = G( d_gen_seed ).detach()  # detach to avoid training G on these labels
      d_fake_decision = D(d_fake_data)
      d_fake_error = criterion(d_fake_decision, torch.zeros([batch_size,1]).to(device))  # zeros = fake
      d_fake_error.backward()
      d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

      dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

    for ii in range(20):  # train G for 20 steps
      G.zero_grad()

      g_gen_seed = (torch.FloatTensor( torch.randn(batch_size,latent_dim ))).to(device)
      g_fake_data = G( g_gen_seed )
      dg_fake_decision = D(g_fake_data)
      g_error = criterion(dg_fake_decision, torch.ones([batch_size,1]).to(device))  # Train G to pretend it's genuine

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
      if FLAGS.model == 'helix':
        t=np.arange(-1.,1,0.1)
        ax.plot(4*np.cos(t*2*np.pi),4*np.sin(t*2*np.pi),4*t, 'r-')
      if FLAGS.model == 'bike_accident':
        t=np.arange(-1.,1,0.1)
        r=4
        ax.plot(r * np.sin(t*2*np.pi),r * np.cos(t*2*np.pi),0*t,'r-')
        ax.plot(r+r * np.sin(t*2*np.pi),0*t,r * np.cos(t*2*np.pi),'r-')
      if FLAGS.model == 'saddle_point':
        x = np.arange(-2,2,0.1)
        y = np.arange(-2,2,0.1)
        x, y = np.meshgrid(x,y)
        z = x**2 - y**2
        ax.plot_wireframe(x,y,z, rstride=5, cstride=5, color="red")
      if FLAGS.model == 'full_cube':
        ax.add_collection3d(Poly3DCollection(face_cube, facecolors='cyan', linewidths=1, edgecolors='k', alpha=.25))

      # plot generated data
      ax.plot(g_fake_data[:,0],g_fake_data[:,1],g_fake_data[:,2],'b.')
      plt.draw()
      plt.title('Inference with PyTorch')
      plt.pause(0.001)
  plt.show()
  if FLAGS.save == True :
    filename = "model_gan_"+FLAGS.model+".pth"
    torch.save({
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'model_type': FLAGS.model
        },filename)
    print('model saved in '+filename)
  if FLAGS.onnx == True:
    loader = importlib.util.find_spec('onnx')
    if loader:
      loader.loader.load_module() # import onnx
      onnx_filename = "model_G_"+FLAGS.model+".onnx"
      x = (torch.FloatTensor(torch.randn(batch_size,latent_dim))).to(device)
      torch.onnx.export(G,x,onnx_filename,export_params=True,verbose=False, input_names=[ "actual_input" ], output_names=[ "output" ])
      print('ONNX export saved in '+onnx_filename)
    else:
      print('module ONNX not installed: pip/conda install onnx')

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_enum('model', 'helix', ['helix','bike_accident','saddle_point','full_cube'], "")
    flags.DEFINE_integer('epochs', 3000, "")
    flags.DEFINE_integer('latent_dim', 3, "")
    flags.DEFINE_bool('save', False, "")
    flags.DEFINE_bool('onnx', False, "to export the model in ONNX format")
    app.run(main)
