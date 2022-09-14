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

class Generator(nn.Module):
  def __init__(self, sz_latent,sz_hidden,sz_out=2):
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(sz_latent,sz_hidden)
    self.fc2 = nn.Linear(sz_hidden,sz_hidden)
    self.fout = nn.Linear(sz_hidden,sz_out)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fout(x)
    return x

def main(argv):
    N_data= FLAGS.N_data
    all_model  = torch.load(FLAGS.model_path, map_location=device)
    size_out   = all_model['G_state_dict']['fout.weight'].shape[0]
    size_latent= all_model['G_state_dict']['fc1.weight'].shape[0]
    latent_dim = all_model['latent_dim']

    G = Generator(latent_dim,size_latent,size_out).to(device)
    G.load_state_dict(all_model['G_state_dict'])
    G.eval()

    gen_seed = (torch.FloatTensor(torch.randn(N_data,latent_dim))).to(device)
    fake_data = G( gen_seed ).detach().to("cpu")
    plt.cla()
    plt.plot(fake_data[:,0],fake_data[:,1],'b.')
    # plt.draw()
    plt.show()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    # flags.DEFINE_enum('model', 'circle', ['circle', 'simple_sin', 'double_sin', 'unbalanced_xor'], "")
    flags.DEFINE_string('model_path',None,'path to the model file')
    flags.DEFINE_integer('N_data', 100, "number of point to generate")
    app.run(main)
