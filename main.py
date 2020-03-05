from __future__ import print_function
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from PIL import Image
# root directory for dataset
dataroot = 'data' # For dataset.Imagefolder there should be a subfolder in the folder
# number of workers for dataloader
workers = 2
#BAtch size during training
batch_size = 4
#image size
img_size = 64
# Number of channel in the training images
nc = 3
# size of z latent vector (i.e size of generator input)
nz = 100
# size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

# num_epochs = 5
num_epochs = 5
 # learning rate for optimizer
lr =0.0002

# Beta Hyper parameter for Adam optimizer
beta1 = 0.5
# Number of GPU
ngpu = 0

#------------------------
# Weight initialization
def Weight_init(W):
    Class_name = W.__class__.__name__
    if Class_name.find('Conv')!=-1:
        nn.init.normal_(W.weight.data,0.0, 0.02)
    if Class_name.find('BatchNorm')!=-1:
        nn.init.normal_(W.weight.data,1.0, 0.02)
        nn.init.constant_(W.bias.data,0)



# We only need images not target
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                           ]))

dataloader = DataLoader(dataset,batch_size =batch_size,shuffle=True,num_workers=workers)

# deside which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >0) else
                      "cpu")
# Plot some training images
real_batch = next(iter(dataloader))
#plt.figure(figsize=(2,2))
#plt.axis("off")
#plt.title("Training Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:4],padding=2,normalize=True).cpu(),(1,2,0)))
#plt.show()

#Creat a Generator Class

class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator, self).__init__()
        self.de_conv1 = self._make_layer_decode(nz,ngf*8,1,0)
        # state size. (ngf*8) x 4 x 4
        self.de_conv2 = self._make_layer_decode(ngf * 8, ngf * 4, 2,1)
        # state size. (ngf*4) x 8 x 8
        self.de_conv3 = self._make_layer_decode(ngf * 4, ngf * 2, 2,1)
        # state size. (ngf*2) x 16 x 16
        self.de_conv3 = self._make_layer_decode(ngf*2,ngf,2,1)
        # state size. (ngf) x 32 x 32
        self.de_conv4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),nn.Tanh())

    def forward(self,input):
        feature_1 = self.de_conv1(input)
        feature_2 = self.de_conv2(feature_1)
        feature_3 = self.de_conv3(feature_2)
        feature_4 = self.de_conv3(feature_3)


    def _make_layer_decode(self,in_nc,out_nc,strid_nm,Padd_nm):
        block=[nn.ConvTranspose2d( in_nc, out_nc, 4, strid_nm, Padd_nm, bias=False),
               nn.BatchNorm2d(out_nc),
               nn.ReLU(True)]
        return nn.Sequential(*block)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(Weight_init)

# Print the model
print(netG)


# Define Discriminator Class
class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()





