import torch
from torchvision import datasets, transforms

### download and prepare
trans = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize( (0.1307,),(0.3081,))])

train_set = datasets.MNIST( './data', train=True, transform=trans, download=True )
test_set = datasets.MNIST( './data', train=False, transform=trans, download=True )

import matplotlib.pyplot as plt

### display some images
# for an alternative see https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def imshow(tensor, title=None):
    img = tensor.cpu().clone()
    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(0.5)

plt.figure()
for ii in range(10):
    # imshow(train_set.TODO , title='MNIST example ({})'.format(train_set.TODO) )
    imshow(train_set.data[ii,:,:] , title='MNIST example ({})'.format(train_set.targets[ii]))


### display colorized images
def imshowcolor(in_tensor, title=None):
    img = in_tensor.cpu().clone()
    img = img.squeeze()
    plt.imshow(img.permute(1, 2, 0))
    if title is not None:
        plt.title(title)
    plt.pause(0.5)

class gray2color_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_in):
        self.dataset_in = dataset_in
    def __len__(self):
        return len(self.dataset_in)
    def __getitem__(self, idx):
        x = self.dataset_in.data[idx]
        image = torch.zeros(3,x.shape[0],x.shape[1])
        label = self.dataset_in.targets[idx]
        if label<3:
            image[0,:,:] = x
        elif label <7:
            image[1,:,:] = x
        else:
            image[2,:,:] = x
        return (image, label)

color_dataset = gray2color_dataset(train_set)
for ii in range(20):
    imshowcolor(color_dataset[ii][0] , title='MNIST color example ({})'.format(color_dataset[ii][1]))

class gray2gliter_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_in):
        self.dataset_in = dataset_in
    def __len__(self):
        return len(self.dataset_in)
    def __getitem__(self, idx):
        x = self.dataset_in.data[idx]
        noise_a  = torch.randn(x.size())
        noise_b  = torch.randn(x.size())
        image = torch.zeros(3,x.shape[0],x.shape[1])
        label = self.dataset_in.targets[idx]
        if label<3:
            image[0,:,:] = x
            image[1,:,:] = x.mul(noise_a)
            image[2,:,:] = x.mul(noise_b)
        elif label <7:
            image[0,:,:] = noise_a
            image[1,:,:] = x
            image[2,:,:] = noise_b
        else:
            image[0,:,:] = noise_a
            image[1,:,:] = x.mul(noise_b)
            image[2,:,:] = x
        return (image, label)

gliter_dataset = gray2gliter_dataset(train_set)
for ii in range(20):
    imshowcolor(gliter_dataset[ii][0] , title='MNIST gliter example ({})'.format(gliter_dataset[ii][1]))

plt.close()

### use dataloader
# batch_size = 100
# train_loader = torch.utils.data.DataLoader(
#                  dataset=train_set,
#                  batch_size=batch_size,
#                  shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#                  dataset=train_set,
#                  batch_size=batch_size,
#                  shuffle=False)