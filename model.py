import torch.nn as nn

'''
# define your own model
class Net(nn.Module):

    # define the layers
    def __init__(self):
        super(Net, self).__init__()

        self.convolution = nn.Sequential(

            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 26 * 26, 1200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1200, 11),
            # nn.Softmax(dim=1)

        )

    def forward(self, x):
        x = self.convolution(x)
        # print(x.shape)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
'''

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