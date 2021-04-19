DLSR_LAB01
===
## Dataset - skewed_food11
- [Food11 dataset download link](https://www.kaggle.com/tohidul/food11)
- Transfer to skewed_foo11 - `build_imbalanced_food11.sh`
    1. copy the script file into the dataset folder `/food11`
    2. run this script file `build_imbalanced_food11.sh`
    
## Modify the code 
- Modify the root path in `main.py` if needed.
    ```=python=
    # Use API to load train dataset
    trainset = torchvision.datasets.ImageFolder(root='food11re/skewed_training/', transform=transform_train)
    
    # Use API to load valid dataset
    validset = torchvision.datasets.ImageFolder(root='food11re/validation', transform=transform_test)
    ```
- Modify the root path in `eval.py` if needed.
    ```=python=
    evalset = datasets.ImageFolder(root='food11re/evaluation/', transform=transform_train)
    ```
## Reference
- [formula-to-compute-the-number-of-macs-in-a-convolutional-neural-network](https://stackoverflow.com/questions/56138754/formula-to-compute-the-number-of-macs-in-a-convolutional-neural-network)
- [Pytorch详解NLLLoss和CrossEntropyLoss](https://blog.csdn.net/qq_22210253/article/details/85229988)
- [PyTorch - 練習kaggle - Dogs vs. Cats - 使用自定義的 CNN model](https://hackmd.io/@lido2370/S1aX6e1nN?type=view)
- [Github: Prakhar998/food-101](https://github.com/Prakhar998/food-101)