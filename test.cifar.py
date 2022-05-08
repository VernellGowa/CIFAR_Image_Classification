
import argparse
import logging
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from models.simple_cnn import SimpleCNN

LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger('matplotlib.font_manager').disabled = True

PATH  =  os.path.dirname(os.path.abspath(__file__))
#DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda')


def get_args_parser():
    parser = argparse.ArgumentParser('Solar panel production prediction', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    return parser.parse_args()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_image(classes, trainloader):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # show images
    imshow(torchvision.utils.make_grid(images))
    
def train(model, epochs, optimizer, criterion, train_loader, test_loader, APPLICATION):

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        idx = 0
        for batch in train_loader:
            optimizer.zero_grad()

            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            prediction = model(X)
            loss = criterion(prediction, y)

            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if idx % 2000 == 1999:    # print every 2000 mini-batches
                logging.info(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

            idx += 1
    end_time = time.time()
    logging.info(f'Finished Training in {end_time - start_time:.2f} seconds')

    #save our trained model:
    #torch.save(model.state_dict(), PATH)

    return model
def check_prediction(classes, model, test_loader):
    model.eval()

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    imshow(torchvision.utils.make_grid(images))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
    

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print(f'Predicted: {predicted}')
            print(f'Labels: {labels}')

def test(model, test_loader):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    logging.info('Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), 100. * correct // len(test_loader.dataset)))
    

def main():
    args = get_args_parser()    
    APPLICATION = "features_{}".format(args.epochs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    batch_size = 32
    data_path = os.path.join(PATH, 'data')

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

    #show_image(classes, train_loader)

    model = SimpleCNN(DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train(model, args.epochs, optimizer, criterion, train_loader, test_loader, APPLICATION)
    test(model, test_loader)


if __name__ == "__main__":
    main()
