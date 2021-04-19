from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from model import Net

if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    evalset = datasets.ImageFolder(root='food11re/evaluation/', transform=transform_train)
    evalloader = DataLoader(dataset=evalset, batch_size=64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net()
    net.load_state_dict(torch.load('model_CNN.pth'))
    net = net.to(device)
    net.eval()

    class_correct = [0. for i in range(11)]
    class_total = [0. for i in range(11)]

    with torch.no_grad():
        for images, labels in tqdm(evalloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = outputs.max(1)
            for i in range(len(predicted)):
                class_total[labels[i]] += 1
                class_correct[labels[i]] += predicted[i].eq(labels[i]).item()

    for i in range(11):
        print('Accuracy of class %2d is %3d/%3d  %.2f%%' % (
        i, class_correct[i], class_total[i], (100 * class_correct[i] / class_total[i])))

    print('Accuracy of the network on the %d test images: %d/%d  %.2f %%'
          % (sum(class_total), sum(class_correct), sum(class_total), (100 * sum(class_correct) / sum(class_total))))
