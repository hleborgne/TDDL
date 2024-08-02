import torch
import numpy as np

from geomloss import SamplesLoss  
loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

aidge_data = torch.tensor(np.load("helix_aidge_data.npy"))
pytorch_data = torch.tensor(np.load("helix_pytorch_data.npy"))

loss(aidge_data,pytorch_data)
