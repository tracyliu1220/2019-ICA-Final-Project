import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
import numpy as np
import random

def printNumber(s, a):
  num = 0
  for i in range(4):
    num *= 10
    num += np.argmax(a[4-i-1])
  print(s + '{:04}'.format(num))

class Trainset(data.Dataset):
  def __init__(self):
    # labels
    f = open('../dataset/ans.txt', 'r')
    lines = f.readlines()
    self.labels = []
    for line in lines:
      line = int(line)
      num = np.array([ 0 for i in range(40) ])
      for i in range(4):
        num[10 * i + line % 10] = 1
        line //= 10
      num = num.astype(np.dtype('float32'))
      self.labels.append(num)  
    # imgs
    self.imgs = []
    for idx in range(10000):
      img = Image.open('../imgs/{:05}.png'.format(idx))
      img = np.array(img)[:,:,0].astype(np.dtype('float32'))
      img = np.where(img < 120, 0, img)
      img = img.reshape(1, 80, 215)
      self.imgs.append(img)
  def __getitem__(self, idx):
    return self.imgs[idx], self.labels[idx]
  def __len__(self):
    return len(self.imgs)

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, (33, 25))
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(10, 30, (5, 5))
    # self.a1 = nn.Linear(15 * 5 * 5, 120);
    self.a1 = nn.Linear(30 * 10 * 45, 120);
    self.a2 = nn.Linear(120, 60)
    self.a3 = nn.Linear(60, 60)
    self.a4 = nn.Linear(60, 40)

  def forward(self, x):
    x = self.pool(func.relu(self.conv1(x)))
    x = self.pool(func.relu(self.conv2(x)))
    # print(x.shape)
    # x = x.view(-1, 15 * 5 * 5)
    x = x.view(x.size(0), -1)
    # print(x.shape)
    x = func.relu(self.a1(x))
    x = func.relu(self.a2(x))
    x = func.relu(self.a3(x))
    x = torch.sigmoid(self.a4(x))
    # x = self.a4(x)
    return x


cnn = CNN()
cnn.load_state_dict(torch.load('cnn-2.pth'))

train_data = Trainset()

while True:
    # idx = int(input())
    _in = input()
    idx = random.randint(10000, 15000)
    # img, label = train_data[idx]
    
    img = Image.open('../imgs/{:05}.png'.format(idx))
    img = np.array(img)[:,:,0].astype(np.dtype('float32'))
    img = np.where(img < 120, 0, img)

    output = cnn(torch.tensor(img.reshape(1, 1, 80, 215)))
    img = img.reshape(80, 215)
    img = Image.fromarray(img)

    # printNumber('label : ', label.reshape((4, 10)))
    printNumber('output: ', output[0].data.numpy().reshape((4, 10)))

    img.show()

