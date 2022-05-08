import torch
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

PATH  =  os.path.dirname(os.path.abspath(__file__))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_image(classes, trainloader, batch_size):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # show images
    imshow(torchvision.utils.make_grid(images))
    x = 1

def main():
    batch_size = 32
    data_path = os.path.join(PATH, 'data')
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ])),
    batch_size=batch_size, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

    show_image(classes, train_loader, batch_size)

if __name__ == "__main__":
    main()