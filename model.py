import torch.nn as nn
import torch.nn.functional as F

# 65%
# define your own model
class Net(nn.Module):

    # define the layers
    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 128, 3)
        self.pool4 = nn.MaxPool2d(2)

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*12*12, 11)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(-1, 128*12*12)
        x = self.drop(x)
        x = self.fc1(x)

        return x


import torch
import torch.nn as nn
from thop import profile

def my_hook_function(self, input, output):

    print("Op:{}".format(str(self.__class__.__name__)))
    for param in self.parameters():
        print("params shape: {}".format(list(param.size())))


from torchsummary import summary

if __name__ == '__main__':

    model = Net()
    input_data = torch.randn(1, 3, 224, 224)
    out = model(input_data)

    summary(model.cuda(), (3, 224, 224))
