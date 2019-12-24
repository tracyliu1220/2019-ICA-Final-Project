import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
import numpy as np
import random
import time

def toNumber(a):
  num = 0
  for i in range(4):
    num *= 10
    num += np.argmax(a[i])
  return num

class Trainset(data.Dataset):
  def __init__(self, n=200000, if_test=False):
    # labels
    if if_test:
        self.n = 10000
        self.type = 'test'
        f = open('../imgs/test/ans.txt', 'r')
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
    else:
        self.n = n
        self.type = 'train'
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
  def __getitem__(self, idx):
    if self.type == 'train':
        img = Image.open('../imgs/{:06}.png'.format(idx))
    else:
        img = Image.open('../imgs/test/{:06}.png'.format(idx))
    img = np.array(img)[:,:,0].astype(np.dtype('float32'))
    # img = np.where(img < 120, 0, img)
    img = img.reshape(1, 80, 215)
    return img, self.labels[idx]
  def __len__(self):
    return self.n

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, (33, 25))
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(10, 30, (5, 5))

    self.drop1 = nn.Dropout(0.2)
    self.drop2 = nn.Dropout(0.2)
    
    self.a1 = nn.Linear(30 * 10 * 45, 120);
    self.a2 = nn.Linear(120, 120)
    self.a3 = nn.Linear(120, 120)
    self.a5 = nn.Linear(120, 40)

  def forward(self, x):
    x = self.pool(func.relu(self.conv1(x)))
    x = self.pool(func.relu(self.conv2(x)))
    x = x.view(x.size(0), -1)
    x = func.relu(self.drop1(self.a1(x)))
    x = func.relu(self.a2(x))
    x = func.relu(self.drop2(self.a3(x)))
    x = torch.sigmoid(self.a5(x))
    return x


cnn = CNN()
cnn.load_state_dict(torch.load('cnn.pth'))
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.BCELoss()
# optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
epoch = 100


test_data = Trainset(if_test=True)
testloader = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)


def Test():
    cnn.eval()
    acc = 0
    acc_d = 0
    acc_in = 0
    t = 0
    for _data in testloader:
        inputs, labels = _data
        outputs = cnn(inputs)

        labels_n = toNumber(labels[0].reshape((4, 10)))
        outputs_n = toNumber(outputs[0].data.numpy().reshape((4, 10)))
        if labels_n == outputs_n:
            acc += 1
        
        cnt = [ 0 for i in range(10) ]

        for i in range(4):
            labels_d = labels_n % 10
            outputs_d = outputs_n % 10
            labels_n //= 10
            outputs_n //= 10
            cnt[labels_d] += 1
            if labels_d == outputs_d:
                acc_d += 1

        outputs_n = toNumber(outputs[0].data.numpy().reshape((4, 10)))
        for i in range(4):
            outputs_d = outputs_n % 10
            outputs_n //= 10
            if cnt[outputs_d]:
                acc_in += 1
        t += 1

    print('\33[96m', end='')
    print('accuracy      :', acc / len(test_data))
    print('digit accuracy:', acc_d / len(test_data))
    print('digit in range:', acc_in / len(test_data))
    print('\33[0m', end='')

def Train():
    cnn.train()
    prev_loss = 100
    for e in range(epoch):
        N = min(max(((e)//3+1)*10000, 10000), 200000)
        train_data = Trainset(n=N)
        trainloader = data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=2)

        learning_rate = 0.0005
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
        running_loss = 0.0
        print('\33[43m', 'epoch:', e, '\33[0m', end='')
        print('  traindata: ', len(train_data))
        for i, _data in enumerate(trainloader, 0):
            inputs, labels = _data
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                avg_loss = running_loss / 100
                if avg_loss < prev_loss:
                    print('\33[104m', i, running_loss / 100, '\33[0m')
                else:
                    print('\33[105m', i, running_loss / 100, '\33[0m')
                running_loss = 0.0
                prev_loss = avg_loss
                idx = random.randint(0, 9)
                labels_n = toNumber(labels[idx].reshape((4, 10)))
                outputs_n = toNumber(outputs[idx].data.numpy().reshape((4, 10)))

                if labels_n == outputs_n:
                    print('\33[92m', end='')
                print('label :', '{:04}'.format(labels_n))
                print('output:', '{:04}'.format(outputs_n))
                print('\33[0m', end='')
                
            if (i+1) % 100 == 0:
                Test()
                cnn.train()
                # torch.save(cnn.state_dict(), './backup_test/cnn-test-'+str(e)+'-'+str(i)+'.pth')

# Train()
Test()
