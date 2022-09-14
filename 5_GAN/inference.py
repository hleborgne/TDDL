import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    latent_dim = all_model['G_state_dict']['fc1.weight'].shape[1]

    G = Generator(latent_dim,size_latent,size_out).to(device)
    G.load_state_dict(all_model['G_state_dict'])
    G.eval()

    gen_seed = (torch.FloatTensor(torch.randn(N_data,latent_dim))).to(device)
    fake_data = G( gen_seed ).detach().to("cpu")
    if fake_data.shape[1]==2:
        plt.cla()
        plt.plot(fake_data[:,0],fake_data[:,1],'b.')
        # plt.draw()
        plt.show()
    elif fake_data.shape[1]==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(fake_data[:,0],fake_data[:,1],fake_data[:,2],'b.')
        plt.show()
    else:
        print('!!! output data should be 2D or 3D (here dim={})'.format(fake_data.shape[1]))

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    # flags.DEFINE_enum('model', 'circle', ['circle', 'simple_sin', 'double_sin', 'unbalanced_xor'], "")
    flags.DEFINE_string('model_path',None,'path to the model file')
    flags.DEFINE_integer('N_data', 100, "number of point to generate")
    app.run(main)
