import torch.nn as nn
import torch.nn.functional as F
import torch
from thop import profile
from thop import clever_format
from torchsummary import summary

class Net(nn.Module):

    # define the layers
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.batchNorm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.batchNorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.batchNorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3)
        self.pool4 = nn.MaxPool2d(2)
        self.batchNorm4 = nn.BatchNorm2d(128)

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 12 * 12, 11)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchNorm1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.batchNorm2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.batchNorm3(x)
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.batchNorm4(x)
        x = self.pool4(x)

        x = x.view(-1, 128 * 12 * 12)
        x = self.drop(x)
        x = self.fc1(x)

        return x


def my_hook_function(self, input, output):

    global total_params, total_macs

    op_type = str(self.__class__.__name__)
    print("{:20}".format(op_type), end="")
    print("{:<20}{:<20}".format(str(list(input[0].size())), str(list(output.size()))), end="")

    params = 0
    for param in self.parameters():
        #print("params shape: {}".format(list(param.size())))
        tmp = 1
        for i in list(param.size()):
            tmp *= i

        params += tmp

    print("{:>10}".format(params), end="")
    total_params += params


    if(op_type == "Conv2d"):

        out_c = (list(self.parameters())[0].size())[0]
        in_c = (list(self.parameters())[0].size())[1]
        kermel_w = (list(self.parameters())[0].size())[2]
        kermel_h = (list(self.parameters())[0].size())[3]
        out_w = list(output.size())[2]
        out_h = list(output.size())[3]
        macs = kermel_h * kermel_w * in_c * out_h * out_w*out_c
        print("{:>15}".format(macs))
        total_macs += macs

    elif(op_type == "Linear"):

        in_shape = list(input[0].size())[1]
        out_shape = list(output.size())[1]
        macs = in_shape * out_shape
        print("{:>15}".format(macs))
        total_macs += macs

    else:
        print("{:>15}".format(0))





if __name__ == '__main__':

    total_params = 0
    total_macs = 0

    model = Net()
    for child in model.children():
        child.register_forward_hook(my_hook_function)

    #op_type             input_shape         output_shape            params           macs
    print("{:20}{:20}{:20}{:>10}{:>15}".format("op_type", "input_shape", "output_shape", "params", "macs"))
    # -------------------------------------------------------------------------------------
    for i in range(85):
        print("-", end="")
    print("")

    input_data = torch.randn(1, 3, 224, 224)
    out = model(input_data)

    # -------------------------------------------------------------------------------------
    for i in range(85):
        print("-", end="")
    print("")

    print("Total params: {:>10}".format(total_params))
    print("Total MACs:   {:>10}".format(total_macs))


    #macs, params = profile(model, inputs=(input_data,), verbose=False)
    #macs, params = clever_format([macs, params], "%.3f")
    #print("Total params: " + str(params), "\nTotal MACs: " + str(macs))
    #summary(model.cuda(), (3, 224, 224))

