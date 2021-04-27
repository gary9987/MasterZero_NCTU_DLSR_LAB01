---
tags: 'Github'
---

masterzero-nctu-DLSR-LAB01
===
## Dataset - skewed_food11
- [Food11 dataset download link](https://www.kaggle.com/tohidul/food11)
- Transfer to skewed_foo11 - `build_imbalanced_food11.sh`
    1. copy the script file into the dataset folder `/food11`
    2. run this script file `build_imbalanced_food11.sh`
    
## Modify the code 
- Modify the root path in `main.py` if needed.
    ```python=46
    # Use API to load train dataset
    trainset = torchvision.datasets.ImageFolder(root='food11re/skewed_training/', transform=transform_train)
    
    # Use API to load valid dataset
    validset = torchvision.datasets.ImageFolder(root='food11re/validation', transform=transform_test)
    ```
- Modify the root path in `eval.py` if needed.
    ```python=45
    evalset = datasets.ImageFolder(root='food11re/evaluation/', transform=transform_train)
    ```

## Steps
### Stpe1: Load the datasets using [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/datasets.html#imagefolder)
- set [transforms](https://pytorch.org/vision/stable/transforms.html#compositions-of-transforms): 
    ```python=
    # The transform function for train data
    transform_train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    # The transform function for validation and evaluation data
    transform_test = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    ```
    - Description:
        - RandomRotation(degrees): 隨機選轉degrees範圍內的角度。
        - RandomResizedCrop(size): 隨機大小，随機長寬比裁減原始圖片，最後將圖片resize到size參數。
        - RandomHorizontalFlip(p=0.5): 依據機率p垂直翻轉圖片。
        - ToTensor(): 將PIL Image或者ndarray轉為tensor，並標準化至[0-1]。
            :::warning
            標準化至[0-1]是直接除以255，若自己的ndarray數據範圍不同，需要自行修改。
            :::
        - Normalize(mean, std): 對數據依照通道進行標準化(先減mean再除std)。
    - Reference:
        - [PyTorch 学习笔记（三）：transforms的二十二个方法](https://zhuanlan.zhihu.com/p/53367135)
        - [Github: Prakhar998/food-101](https://github.com/Prakhar998/food-101)

- set the datasets
    ```python=
    # Use API to load train dataset
    trainset = torchvision.datasets.ImageFolder(root='food11re/skewed_training/', transform=transform_train)
    # Use API to load valid dataset
    validset = torchvision.datasets.ImageFolder(root='food11re/validation', transform=transform_test)


    ```
- Load datasets with DataLoader
    ```python=59
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                                  shuffle=True, num_workers=2)

    validloader = torch.utils.data.DataLoader(validset, batch_size=20,
                                                  shuffle=True, num_workers=2)
    ```
### Step2: Build a model
```python=
class Net(nn.Module):

    # define the layers
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.batchNorm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batchNorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.batchNorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3)
        self.batchNorm4 = nn.BatchNorm2d(128)

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 12 * 12, 11)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchNorm1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.batchNorm2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.batchNorm3(x)
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.batchNorm4(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 12 * 12)
        x = self.drop(x)
        x = self.fc1(x)

        return x

'''
op_type             input_shape         output_shape            params           macs
-------------------------------------------------------------------------------------
Conv2d              [1, 3, 224, 224]    [1, 16, 222, 222]          448       21290688
BatchNorm2d         [1, 16, 222, 222]   [1, 16, 222, 222]           32              0
MaxPool2d           [1, 16, 222, 222]   [1, 16, 111, 111]            0              0
Conv2d              [1, 16, 111, 111]   [1, 32, 109, 109]         4640       54747648
BatchNorm2d         [1, 32, 109, 109]   [1, 32, 109, 109]           64              0
MaxPool2d           [1, 32, 109, 109]   [1, 32, 54, 54]              0              0
Conv2d              [1, 32, 54, 54]     [1, 64, 52, 52]          18496       49840128
BatchNorm2d         [1, 64, 52, 52]     [1, 64, 52, 52]            128              0
MaxPool2d           [1, 64, 52, 52]     [1, 64, 26, 26]              0              0
Conv2d              [1, 64, 26, 26]     [1, 128, 24, 24]         73856       42467328
BatchNorm2d         [1, 128, 24, 24]    [1, 128, 24, 24]           256              0
MaxPool2d           [1, 128, 24, 24]    [1, 128, 12, 12]             0              0
Dropout             [1, 18432]          [1, 18432]                   0              0
Linear              [1, 18432]          [1, 11]                 202763         202752
-------------------------------------------------------------------------------------
Total params:     300683
Total MACs:    168548544
'''
```

### Step3: Loss function and optimizer
- Loss function: CrossEntropy
- Optimize: Adam
```python=
# loss function
criterion = nn.CrossEntropyLoss()
# optimization algorithm
optimizer = optim.Adam(net.parameters())
```

### Step4: Train
- 100 epochs
- 每一個mini-batch都計算validation set的loss，儲存valid loss最小的model。

### Step5: Evaluation
- Function for show top k accuracy
    ```python=
    def evaluteTopK(k, model, loader):
        model.eval()

        class_correct = [0. for i in range(11)]
        class_total = [0. for i in range(11)]

        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                y_resize = labels.view(-1, 1)
                _, predicted = outputs.topk(k, 1, True, True)

                for i in range(len(predicted)):
                    class_total[labels[i]] += 1
                    #print(torch.eq(predicted[i], y_resize[i]).sum().float().item())
                    class_correct[labels[i]] += torch.eq(predicted[i], y_resize[i]).sum().float().item()

        for i in range(11):
            print('Top %d Accuracy of class %2d is %3d/%3d  %.2f%%' % (
                k, i, class_correct[i], class_total[i], (100 * class_correct[i] / class_total[i])))

        print('Top %d accuracy of the network on the %d test images: %d/%d  %.2f %%'
              % (k, sum(class_total), sum(class_correct), sum(class_total), (100 * sum(class_correct) / sum(class_total))))

        return 100 * sum(class_correct) / sum(class_total)
    ```
- Load evaluation set and evaluate
    ```python=
    transform_train = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    evalset = datasets.ImageFolder(root='food11re/evaluation/', transform=transform_train)
    evalloader = DataLoader(dataset=evalset, batch_size=100)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net()
    net.load_state_dict(torch.load('model_CNN.pth'))
    net = net.to(device)
    net.eval()

    print(evaluteTopK(1, net, evalloader))
    print(evaluteTopK(3, net, evalloader))

    ```
## Reference
- [formula-to-compute-the-number-of-macs-in-a-convolutional-neural-network](https://stackoverflow.com/questions/56138754/formula-to-compute-the-number-of-macs-in-a-convolutional-neural-network)
- [Pytorch详解NLLLoss和CrossEntropyLoss](https://blog.csdn.net/qq_22210253/article/details/85229988)
- [PyTorch - 練習kaggle - Dogs vs. Cats - 使用自定義的 CNN model](https://hackmd.io/@lido2370/S1aX6e1nN?type=view)
- [Github: Prakhar998/food-101](https://github.com/Prakhar998/food-101)