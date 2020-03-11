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
from spectral_normalization import SpectralNorm
from IPython.display import HTML

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
lr = 0.0002

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
        self.de_conv4 = self._make_layer_decode(ngf*2,ngf,2,1)
        # state size. (ngf) x 32 x 32
        self.de_conv5 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),nn.Tanh())

    def forward(self,input):
        feature_1 = self.de_conv1(input)
        feature_2 = self.de_conv2(feature_1)
        feature_3 = self.de_conv3(feature_2)
        feature_4 = self.de_conv4(feature_3)
        feature_5 = self.de_conv5(feature_4)
        return feature_5


    def _make_layer_decode(self,in_nc,out_nc,strid_nm,Padd_nm):
        block=[nn.ConvTranspose2d(in_nc, out_nc, 4, strid_nm, Padd_nm, bias=False),
               nn.BatchNorm2d(out_nc),
               nn.ReLU(True)]
        return nn.Sequential(*block)


# Create the generator
NetG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    NetG = nn.DataParallel(NetG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
NetG.apply(Weight_init)

# Print the model
print(NetG)


# Define Discriminator Class
class Discriminator(nn.Module):
    def __init__(self,ngpu,Norm_pm):
        super(Discriminator,self).__init__()
        self.Norm_pm = Norm_pm
        self.ngpu = ngpu
        # input is (nc) x 64 x 64
        if Norm_pm =='BatchNorm':
          self.encode1 = nn.Sequential(nn.Conv2d(nc,ndf,4,2,1,bias=False),nn.LeakyReLU(0.2,inplace=True))
          # state size. (ndf) x 32 x 32
          self.encode2 = self._make_layer_encode(ndf,ndf*2,2,1)
          # state size. (ndf*2) x 16 x 16
          self.encode3 = self._make_layer_encode(ndf*2,ndf*4,2,1)
          # state size. (ndf*4) x 8 x 8
          self.encode4 = self._make_layer_encode(ndf * 4, ndf*8, 2, 1)
          # state size. (ndf*8) x 4x 4
          self.encode5 = nn.Sequential(nn.Conv2d(ndf*8,1,4,1,0,bias=False),nn.Sigmoid())
        else:
          self.encode1 = SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
          # state size. (ndf) x 32 x 32
          self.encode2 = SpectralNorm(nn.Conv2d(ndf, ndf * 2,4, 2, 1,bias=False))
          # state size. (ndf*2) x 16 x 16
          self.encode3 = SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 2, 1,bias=False))
          # state size. (ndf*4) x 8 x 8
          self.encode4 = SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 2, 1,bias=False))
          # state size. (ndf*8) x 4x 4
          self.encode5 =SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())


    def forward(self,input):
          feature_1 = self.encode1(input)
          feature_2 = self.encode2(feature_1)
          feature_3 = self.encode3(feature_2)
          feature_4 = self.encode4(feature_3)
          feature_5 = self.encode5(feature_4)
          return feature_5



    def _make_layer_encode(self,in_nc,out_nc,stride_nm,Padd_nm):
        block =[nn.Conv2d(in_nc,out_nc,4,stride_nm,Padd_nm,bias=False),
                nn.BatchNorm2d(out_nc),
                nn.LeakyReLU(True)]
        return nn.Sequential(*block)

NetD = Discriminator(ngpu,'BatchNorm').to(device)
if (device.type=='cuda') and (ngpu>0):
    NetD = nn.DataParallel(NetD,list(range(ngpu)))

NetD.apply(Weight_init)

    # Print the model
print(NetD)

#-------------Define loss funtion
criterion = nn.BCELoss()
# Create a batch of noise

fixed_noise = torch.randn(4, nz, 1, 1, device=device)

lable_re = 1
lable_fa = 0

Optimizer_D = optim.Adam(NetD.parameters(),lr=lr, betas=(beta1, 0.999))
Optimizer_G = optim.Adam(NetG.parameters(),lr=lr, betas=(beta1, 0.999))

# Training Procedure ---------------
G_loss = []
D_Loss = []
img_G_list = []
iters = 0
#-------------
print("Starting Training.........")
# For each epoch
for epoch in range(num_epochs):
   # For each batch in dalaloader
    for i,data in enumerate(dataloader,0):
        # Update Discriminator
        # Train Discriminator with Real batches
        NetD.zero_grad()
        Real_data_cpu = data[0].to(device)
        batch_size = Real_data_cpu.size(0)
        label = torch.full((batch_size,),lable_re,device=device)
        output_D = NetD(Real_data_cpu).view(-1)
        #Calculate loss for real data batch
        Err_real_D = criterion(output_D,label)
        #computes dErr_real_D/dx for every parameter x
        Err_real_D.backward()
        # Average loss
        D_x = output_D.mean().item()
        # Train discriminator for fake batches
        noise = torch.randn(batch_size,nz,1,1,device=device)
        # Generate fake image using generator
        fake_img = NetG(noise)
        label.fill_(lable_fa)
        output_D = NetD(fake_img.detach()).view(-1)
        Err_fake_D = criterion(output_D,label)
        Err_fake_D.backward()
        D_G_z1 = output_D.mean().item()
        ErrD = Err_fake_D+Err_real_D
        # Update D
        Optimizer_D.step()
        # Update Generator
        NetG.zero_grad()
        label.fill_(lable_re)
        output_D_G = NetD(fake_img).view(-1)
        # Claculate Generator loss
        ErrG = criterion(output_D_G,label)
        ErrG.backward()
        D_G_z2 = output_D_G.mean().item()
        Optimizer_G.step()
        #-----------output status--------------
        if i%5==0:
            print('\Epoc: %d \Num_epoc: %d \Err_D: %.4f Err_G: %.4f' %(epoch,num_epochs,ErrG.item(),ErrD.item()))
        G_loss.append(ErrG.item())
        D_Loss.append((ErrD.item()))
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = NetG(fixed_noise).detach().cpu()
            img_G_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# Plot the results
plt.figure(figsize=(10,5))
plt.title("Generator and discriminator loss (TRaining phase)")
plt.plot(G_loss,label="G")
plt.plot(D_Loss,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()




#%%capture
fig = plt.figure(figsize=(2,2))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_G_list]
plt.show()

ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

























