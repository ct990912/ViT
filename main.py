from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import *

device = torch.device('cuda')
cifar10_path = r"D:\Train_Data\cifar10"


def get_set(path):
    train_transform = transforms.Compose(
        [transforms.RandomCrop((32, 32), padding=2), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10(path, True, train_transform)
    val_set = CIFAR10(path, False, val_transform)
    return train_set, val_set


def train(lr=0.01):
    print('train start')
    train_iter_n = 1
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for epoch_n in range(epoch):
        total_train_acc = 0
        total_val_acc = 0
        if ((epoch_n + 1) % 60 == 0):
            lr /= 10
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = net(x)
            loss = loss_fn(pred, y)
            train_acc = np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == y.cpu().numpy())
            total_train_acc += train_acc
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss, train_iter_n)
            writer.add_scalar('train_acc', train_acc / pred.size()[0], train_iter_n)
            train_iter_n += 1
        with torch.no_grad():
            net.eval()
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = net(x)
                val_acc = np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == y.cpu().numpy())
                total_val_acc += val_acc
        print('----------------epoch %d is finished----------------' % (epoch_n + 1))
        print("train acc ", total_train_acc / len(train_set))
        print("val acc ", total_val_acc / len(val_set))
        writer.add_scalar('total_train_acc', total_train_acc / len(train_set), epoch_n + 1)
        writer.add_scalar('total_val_acc', total_val_acc / len(val_set), epoch_n + 1)
        torch.save(net.state_dict(), 'logs/net.pth')


if __name__ == '__main__':
    epoch = 160
    writer = SummaryWriter("logs")
    net = ViT(32, 4, 3, 10, 128, 3, 8, 4).to(device)
    train_set, val_set = get_set(cifar10_path)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=4)
    train()
