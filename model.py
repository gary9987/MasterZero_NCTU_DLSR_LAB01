import torch.nn as nn

# 65%
# define your own model
class Net(nn.Module):

    # define the layers
    def __init__(self):
        super(Net, self).__init__()

        self.flatten = nn.Flatten()
        self.convolution = nn.Sequential(

            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        self.linear_relu_stack = nn.Sequential(

            nn.Dropout(p=0.5),
            nn.Linear(128 * 12 * 12, 11)

        )

    def forward(self, x):
        x = self.convolution(x)
        #print(x.shape)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


import torch
import torch.nn as nn


def my_hook_function(self, input, output):
    print("Op:{}".format(str(self.__class__.__name__)))
    for param in self.parameters():
        print("params shape: {}".format(list(param.size())))



if __name__ == '__main__':

    model = Net()
    model.convolution.register_forward_hook(my_hook_function)
    model.flatten.register_forward_hook(my_hook_function)
    model.linear_relu_stack.register_forward_hook(my_hook_function)
    input_data = torch.randn(1, 3, 224, 224)
    out = model(input_data)
