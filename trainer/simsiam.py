# -*- coding: utf-8 -*-
"""SimSiam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A2F_n--qXIKjtoNmc4omja6-7n8Pi8Ak
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
from google.colab import drive
drive.mount('/content/drive')

!wget https://github.com/milvus-io/milvus/releases/download/v2.2.13/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/
# %ls
!bash install_docker_in_colab.sh

!sudo whoami

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/milvus
!docker --version
#%sudo service docker start
!sudo service docker start
#!sudo systemctl start docker
!sudo docker compose up -d --privileged=true

# Commented out IPython magic to ensure Python compatibility.
# %ls
!./scripts/install_deps.sh
!make
# To build GPU version, add -g option, and switch the notebook settings with GPU
#((Edit -> Notebook settings -> select GPU))
# !./build.sh -t Release -g

import numpy as np

a = np.zeros((3,2))
print(*a.shape)
a[1,1] = 1
a[2,1] = 2
print(a)
#print(a)
b = np.random.rand(*a.shape)
c = b<0.9
print(c)
print(np.where((c==True)&(a==2)))
#d = np.invert(a, where = c)
print(d.astype(int))

from google.colab import auth
auth.authenticate_user()

!export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
!echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

!echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
!apt -qq update
!apt -qq install gcsfuse

!mkdir wmap-811k
!gcsfuse --implicit-dirs wmap-811k wmap-811k

!pip install umap-learn[plot]
!pip install holoviews
!pip install -U ipykernel
!pip install umap_learn

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

!ls

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class BinaryNoiseGeneration:
  """Randomly generate noise"""
  def __init__(self, p=0.1):
    assert isinstance(p, float)
    self.p = p

  def __call__(self, img):
    assert isinstance(img, np.ndarray)
    #img = img.astype(bool)
    #print(a)
    img_c = img.copy()
    mask = np.random.rand(*img_c.shape)<self.p
    img_c[np.where((img_c==2)&(mask))] = 1
    img_c[np.where((img_c==1)&(mask))] = 2
    #out = np.multiply(img, where = mask).astype(np.uint8)
    return img_c

class noiseSmooth:
  """smooth noise for low density pattern"""

import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils import data



file = open('wmap-811k/WM811K.pkl', 'rb')
X = pickle.load(file)

X = X[X.failureType!='none']

X.failureType.astype(str).unique()

np.unique(X.waferMap.tolist()[0])

class wafermap(Dataset):

    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
            #y = self.transform(y)

        return x, y

x_train = X[X.trainTestLabel == 'Training'].waferMap.tolist()
y_train = X[X.trainTestLabel == 'Training'].failureType.tolist()
transform = TwoCropsTransform(transforms.Compose([BinaryNoiseGeneration(p=0.05), transforms.ToPILImage(),transforms.Resize((224,224)), transforms.RandomRotation(degrees=(0,360)),transforms.ToTensor()]))
waferdata = wafermap(X=x_train, Y=y_train, transform = transform)

x_test = X[X.trainTestLabel=='Testing'].waferMap.tolist()
y_test = X[X.trainTestLabel=='Testing'].failureType.tolist()
transform = TwoCropsTransform(transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()]))
waferdata_test = wafermap(X=x_test, Y=y_test, transform=transform)

train_loader = torch.utils.data.DataLoader(
    waferdata,
    batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    waferdata_test,
    batch_size=32, shuffle=True
)

for img, target in train_loader:
  print(img[0].shape)
  print(target[0])
  plt.imshow(np.asarray(img[0][0][0]))

  plt.show()
  plt.imshow(np.asarray(img[1][0][0]))
  plt.show()
  #for img in rotated_imgs:
  #  plt.imshow(np.squeeze(img))
  break

a = models.__dict__['resnet18']

b= a(num_classes=64)

def get_output_size(w, k, p=0, s=1):
  return 1+((w-k+2*p)/s)

base_model = models.resnet18()
base_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

class EMA():
  def __init__(self,alpha):
    super().__init__()
    self.alpha = alpha

  def update_average(self, old,new):
    if old is None:
      return new
    return old*self.alpha + (1-self.alpha)*new

def byol_loss_fn(x, y):
  x = F.normalize(x, dim=-1, p=2)
  y = F.normalize(x, dim=-1, p=2)
  return 2-2*(x*y).sum(dim=-1)

class MLP(nn.Module):
  def __init__(self, dim, embedding_size:str, hidden_size:str, batch_norm_mlp=True):
    super().__init__()
    norm = nn.BatchNorm1d(hidden_size)
    self.net=nn.Sequential(
        nn.Linear(dim, hidden_size),
        norm,
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, embedding_size)
    )
  def forward(self,x):
    return self.net(x)

class AddProjHead(nn.Module):
  def __init__(self, model, in_features, layer_name, hidden_size, embedding_size, batch_norm_mlp=True):
    super(AddProjHead, self).__init__()
    self.backbone = model
    setattr(self.backbone, layer_name, nn.Identiy())
    self.backbone.conv1=torch.nn.Conv2d(3)

class byol(nn.Module):
  def __init__(self,
               base:nn.Module,
               #batch_norm_mlp = True,
               input_dim:int,
               internal_dim:int,
               pred_dim:int,
               ma_decay:int=0.99
              ):
    super(byol, self).__init__()
    self.student_model=base
    self.teacher_model=self._get_teacher()
    self.target_ema_updater=EMA(ma_decay)
    self.projector = nn.Sequential(
    nn.Linear(input_dim, internal_dim, bias = False),
    nn.BatchNorm1d(internal_dim),
    nn.ReLU(),
    nn.Linear(internal_dim,pred_dim))

  @torch.no_grad()
  def _get_teacher(self):
    return copy.deepcopy(self.student_model)

  @torch.no_grad()
  def update_moving_average(self):
    for student_params, teacher_params in zip(self.student_model.parameters(),self.teacher_model.parameters()):
      old_w, up_w = teacher_params.data, student_params.data
      teacher_params.data = self.target_ema_updater.update_average(old_w, up_w)

  def forward(self, x1, x2):
    student_proj_one = self.student_model(x1)
    student_proj_two = self.student_model(x2)
    student_pred_one = self.projector(student_proj_one)
    student_pred_two = self.projector(student_proj_two)
    with torch.no_grad():
      teacher_proj_one=self.teacher_model(x1).detach()
      teacher_proj_two=self.teacher_model(x2).detach()
    loss_one = byol_loss_fn(student_pred_one, teacher_proj_two)
    loss_two = byol_loss_fn(student_pred_two, teacher_proj_one)
    return student_pred_one,student_pred_two,  teacher_proj_one,teacher_proj_two
  def name(self):
    return 'byol'

class base_deep(nn.Module):
    def __init__(self):
        super(base_deep, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #torch.nn.Dropout(p=1 - keep_prob)
            )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #torch.nn.Dropout(p=1 - keep_prob)
            )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #torch.nn.Dropout(p=1 - keep_prob))
        )
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            #torch.nn.Dropout(p=1 - keep_prob)
            )
        self.fc1 = nn.Linear()
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

from collections import OrderedDict
# build 2 layer model

class base(nn.Module):
  def __init__(self, imgChannels=1, featureDim=32*20*20, zDim=256, zDim2=16, kernel_size=5) -> None:
    super(base, self).__init__()
    #building blocks
    self.encConv1=nn.Conv2d(in_channels=imgChannels, out_channels=16, kernel_size=kernel_size)
    self.encConv2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size)
    self.encConv3=nn.Conv2d(in_channels= 32, out_channels=64, kernel_size=kernel_size )
    self.encConv4=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size)
    #self.encFC1=nn.Linear(in_features=128*12*12, out_features=zDim, bias = False)
    #self.encFC11 = nn.Linear(in_features=zDim, out_features = zDim)
    #self.encFC13 = nn.Linear(in_features= zDim, out_features= zDim)
    self.encFC2=nn.Linear(in_features=zDim, out_features=zDim2, bias = False)
    self.featureDim=128*12*12
    #self.classification_layer=nn.Linear(in_features=zDim, out_features=n_class)
    self.conv_net = nn.Sequential(self.encConv1,
                                  nn.ReLU(),
                                  self.encConv2,
                                  nn.ReLU(),
                                  self.encConv3,
                                  nn.ReLU(),
                                  self.encConv4,
                                  nn.ReLU())
    self.fc_net = nn.Sequential(nn.Linear(in_features=128*12*12, out_features=zDim, bias = False),
                                nn.BatchNorm1d(zDim),
                                nn.ReLU(),
                                #self.encFC1,
                                #nn.BatchNorm1d(zDim),
                                #nn.ReLU(),
                                #self.encFC11,
                                #nn.ReLU(),
                                #self.encFC13,
                                #nn.ReLU(),
                                self.encFC2,
                                nn.BatchNorm1d(zDim2))
  def forward(self, x):
    conv_net = nn.Sequential(self.encConv1, nn.ReLU(), self.encConv2, nn.ReLU())
    return self.fc_net(self.conv_net(x).view(-1, self.featureDim))


class simSiam(nn.Module):
  def __init__(self, base, dim =16, pred_dim=4):
    super(simSiam, self).__init__()
    self.base = base
    self.projector = nn.Sequential(
        nn.Linear(dim, pred_dim, bias = False),
        nn.BatchNorm1d(pred_dim),
        nn.ReLU(),
        nn.Linear(pred_dim,dim))
  def forward(self, x1, x2):
    z1 = self.base(x1)
    z2 = self.base(x2)
    p1 = self.projector(z1)
    p2 = self.projector(z2)
    return p1,p2,z1.detach(),z2.detach()

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=128, pred_dim=16):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device

from torch.utils import data

batch_size = 32
train_data = datasets.MNIST('data', train = True,
                            download = True,
                            transform= TwoCropsTransform(transforms.Compose([transforms.Resize((224,224)), transforms.RandomRotation(degrees=(0,360)),transforms.ToTensor()])),
                            )
test_data = datasets.MNIST('data',
                           train = False,
                           download = True,
                           transform= transforms.ToTensor()
                           )
print(type(train_data))
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1)
train_data = datasets.MNIST('data', train = True, download = True, transform=transforms.ToTensor())

train_target = [6,9]

# partial_train_loader = torch.utils.data.DataLoader(
#     select_subset_by_target(train_data, target = train_target),
#     batch_size=batch_size,
#     shuffle=True)



rotater = transforms.RandomRotation(degrees = (0,180))
for img, target in train_loader:
  #print('i',i)
  print(img[0].shape)
  #imgs, _ = data
  #rot_img = transforms.functional.rotate(img, 60)
  #img= np.transpose(imgs[0].cpu(), [1,2,0])
  #img2 = cv2.flip(img, 0)
  #rotated_imgs = [rotater(img) for _ in range(4)]
  plt.imshow(np.asarray(img[0][0][0]))

  plt.show()
  plt.imshow(np.asarray(img[1][0][0]))
  plt.show()
  #for img in rotated_imgs:
  #  plt.imshow(np.squeeze(img))
  break

base_model = models.resnet18

learning_rate = 1e-3
import numpy as np
load_path = '/content/drive/My Drive/simsiam_wmap.pth'
load = False
#model = simSiam(base = base(), dim=16, pred_dim=4).to(device)
model = SimSiam(base_encoder=base_model, dim=1024, pred_dim=256).to(device)
if load:
  model.load_state_dict(torch.load(load_path))
#model = byol(base = base(zDim2=8), input_dim=8,internal_dim=32, pred_dim=8).to(device)
#print(model.name())
criterion = nn.CosineSimilarity(dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epoch = 5
loop = 1000
def train(train_loader, model, criterion, optimizer, epoch):
  model.train()
  losses = []
  for i, (img,_) in enumerate(train_loader):
    #angle = np.random.randint(1,359)
    #rot_img = transforms.RandomRotation(img, angle)
    img = [_img.to(device) for _img in img ]
    #rot_img = rot_img.to(device)
    p1,p2,z1,z2 = model(x1=img[0], x2=img[1])
    #print(p1,p2)
    loss = -(criterion(p1,z2).mean()+criterion(p2,z1).mean())*0.5
    #print(loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #if model.__name__() == "byol":
      #print('in')
    #model.update_moving_average()
  return np.mean(losses)

for i in range(epoch):
  losses = train(train_loader = train_loader, model = model,  criterion = criterion, optimizer = optimizer, epoch = epoch)
  print(losses)

#!pip uninstall umap
#!pip install umap-learn
import numpy
from umap import umap_ as umapp
import umap.plot
def plot_latent(autoencoder, data, num_batches=100):
    count = 20
    z_all = []
    y_all = []
    for i, (x, y) in enumerate(data):
      if i > count:
        break
      #print(x.ToTensor().shape)
      #z = autoencoder.reparameterize(*autoencoder.encoder(x.cuda()))
      #z = autoencoder.predictor(autoencoder.encoder(x[0].cuda()))
      z = autoencoder.encoder(x[0].cuda())
      #z = autoencoder.encoder()
      #z = autoencoder.teacher_model(x.cuda())
      z = z.to('cpu').detach().numpy()
      print(z.shape)
      z_all.append(z)
      y_all.append(y)
    #print(z_all)
    z_all = np.concatenate(z_all)
    y_all = np.concatenate(y_all)
    print(y_all.shape)
    mapper = umapp.UMAP().fit(z_all)
    umap.plot.points(mapper, labels=y_all)
      #break
      #if i > num_batches:
      #  plt.colorbar()
      #  break

def plot_svd(autoencoder, data, num_batches=100):
  s_all = 0
  count = 10
  for i, (x,y) in enumerate(data):
    if i > count:
      break
    z = autoencoder.encoder(x[1].cuda())
    z = z.to('cpu').detach().numpy()
    _,s,_ = np.linalg.svd(z.T.dot(z))
    #plt.plot(range(s.shape[0]), s)
    s_all+=s
    print(s.shape)
  s_all/=i
  plt.plot(range(s_all.shape[0], s_all))

plot_svd(model, data=train_loader)

plot_latent(autoencoder=model, data = train_loader)
plt.show()



torch.save(model.state_dict(), '/content/drive/My Drive/simsiam_wmap.pth')

