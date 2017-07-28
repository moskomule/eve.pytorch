from eve import Eve

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# variables
batch_size = 128
epochs = 100
cuda = torch.cuda.is_available()

# load data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/cifar10', train=True, download=True,
                     transform=transform),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/cifar10', train=False, transform=transform),
    batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.dense1 = nn.Linear(in_features=64 * 25, out_features=512)
        self.dense1_bn = nn.BatchNorm1d(512)
        self.dense2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.dropout(F.max_pool2d(self.conv2(x), 2), 0.25))
        x = F.relu(self.conv3(x))
        x = F.relu(F.dropout(F.max_pool2d(self.conv4(x), 2), 0.25))
        x = x.view(-1, 64 * 25)  # reshape
        x = F.relu(self.dense1_bn(self.dense1(x)))
        return F.log_softmax(self.dense2(x))


def train(epoch, model, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        def closure():
            optimizer.zero_grad()  # reset reset optimizer
            output = model(data)
            loss = F.nll_loss(output, target)  # negative log likelihood loss
            loss.backward()  # backprop
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.data[0] / len(train_loader)
        if batch_idx % 20 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:>4.2%})] Loss: {:>5.3}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       batch_idx / len(train_loader), total_loss),
                end="")
    return total_loss


def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2%})'.format(
        test_loss, correct, len(test_loader.dataset),
        correct / len(test_loader.dataset)))
    return test_loss


def plot(loss_a, loss_b, filename, ylabel):
    import matplotlib
    matplotlib.use("AGG")
    import matplotlib.pyplot as plt
    plt.plot(loss_a)
    plt.plot(loss_b)
    plt.legend(["Eve", "Adam"])
    plt.xlabel("epochs")
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()


print("Eve")
eve_loss = []
eve_test_loss = []
model = Net()
if cuda:
    model.cuda()
optimizer = Eve(model.parameters())
for i in range(1, epochs + 1):
    eve_loss.append(train(i, model, optimizer))
    eve_test_loss.append(test(i, model))

print("Adam")
adam_loss = []
adam_test_loss = []
model = Net()
if cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters())
for i in range(1, epochs + 1):
    adam_loss.append(train(i, model, optimizer))
    adam_test_loss.append(test(i, model))

plot(eve_loss, adam_loss, "eve_loss.png", "training loss")
plot(eve_test_loss, adam_test_loss, "eve_test_loss.png", "testing loss")
