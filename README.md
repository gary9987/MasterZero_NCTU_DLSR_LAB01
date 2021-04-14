DLSR_LAB01
===
## Dataset - skewed_food11
- [Food11 Download link](https://www.kaggle.com/tohidul/food11)
- Transfer to skewed_foo11 - `build_imbalanced_food11.sh`
    1. copy the script file into the dataset folder `/food11`
    2. run this script file `build_imbalanced_food11.sh`
    
## Mod the code 
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