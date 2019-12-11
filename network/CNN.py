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
    num += np.argmax(a[i])
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
      # img = np.where(img < 120, 0, img)
      img = img.reshape(1, 80, 215)
      self.imgs.append(img)
  def __getitem__(self, idx):
    return self.imgs[idx], self.labels[idx]
  def __len__(self):
    return len(self.imgs)

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, (32, 25))
    self.conv1.weight.data = patterns
    self.conv1.weight.requires_grad = False
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(10, 30, (5, 5))
    
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

getPatterns()
while 1:
    pass

cnn = CNN()
cnn.load_state_dict(torch.load('cnn-2.pth.backup'))
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.BCELoss()
# optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.00005)
epoch = 100

train_data = Trainset()
trainloader = data.DataLoader(train_data, batch_size=20, num_workers=2)

print('dataset prepared')

# cnn.train()
for e in range(epoch):
    running_loss = 0.0
    print('epoch:', e)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # print(inputs)
        
        optimizer.zero_grad()

        outputs = cnn(inputs)

        # print(outputs)
        # print(labels)
        loss = loss_func(outputs, labels)
        # print(loss.item())
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        # print(loss.item())
        # print(i, loss.item() / 100)
        if (i+1) % 100 == 0:
            print(i, running_loss / 100)
            running_loss = 0.0
            idx = random.randint(0, 19)
            printNumber('label : ', labels[idx].reshape((4, 10)))
            printNumber('output: ', outputs[idx].data.numpy().reshape((4, 10)))
            # print(labels[idx].reshape((4, 10)))
            # print(outputs[idx].data.numpy().reshape((4, 10)))
    # torch.save(cnn.state_dict(), './cnn-2.pth')




