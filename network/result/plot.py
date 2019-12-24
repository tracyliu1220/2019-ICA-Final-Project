import numpy as np
import matplotlib.pyplot as plt
        
f = open('test_result.txt', 'r')
lines = f.readlines()
data = []

# testdata(0), loss(1), epoch(2)
# test:  acc(3), combination acc(4), digit acc(5), digit in range(6)
# train: acc(7), combination acc(8), digit acc(9), digit in range(10)

for line in lines:
    data.append(line.split())

x          = np.array([ i for i in range(1, 64) ])
testdata_n = np.array([   int(data[i][0]) for i in range(0, 63) ])
loss       = np.array([ float(data[i][1]) for i in range(0, 63) ])
test_acc   = np.array([ float(data[i][3]) for i in range(0, 63) ])
train_acc  = np.array([ float(data[i][7]) for i in range(0, 63) ])


fig, ax = plt.subplots(nrows=2, ncols=1)

color = 'tab:red'
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss', color=color)
ax[0].plot(x, loss, color=color)
ax[0].tick_params(axis='y', labelcolor=color)

ax0_twin = ax[0].twinx()

ax0_twin.set_ylabel('accuracy')
ax0_twin.plot(x, train_acc, x, test_acc)
ax0_twin.legend(('train acc', 'test acc'), loc='upper right')

ax[1].set_xlabel('epoch')
ax[1].set_ylabel('# of training data')
ax[1].plot(x, testdata_n)

fig.tight_layout()
plt.show()
