import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data_loader import PoseDataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14, 30)
        self.fc1_bn = nn.BatchNorm1d(30)
        self.fc2 = nn.Linear(30, 20)
        self.fc2_bn = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, 7)

    def forward(self, x):
        x = x.view(-1, 14)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x


def validate(model, criterion, test_loader, device):
    loss = 0

    for idx, (skel_2d, skel_z) in enumerate(test_loader):
        inputs, labels = skel_2d.to(device), skel_z.to(device)
        outputs = model(inputs)
        loss += criterion(outputs, labels).item()

    return loss / len(test_loader)


def train():
    # load data
    pose_dataset = PoseDataset('panoptic_dataset.pickle')

    # random, non-contiguous train/val split
    indices = list(range(len(pose_dataset)))
    val_size = round(len(pose_dataset) * 0.1)
    val_idx = np.random.choice(indices, size=val_size, replace=False)
    train_idx = list(set(indices) - set(val_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset=pose_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset=pose_dataset, batch_size=32, sampler=val_sampler)

    # save val_idx
    np.save('val_idx.npy', val_idx)

    # cpu or gpu?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device {}'.format(device))

    # define net
    net = Net()
    net.to(device)

    # define loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # train
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            skel_2d, skel_z = data
            inputs = skel_2d
            labels = skel_z
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # validate and print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                # validate
                with torch.no_grad():
                    net.eval()
                    val_loss = validate(net, criterion, val_loader, device)
                    net.train()
                    scheduler.step(val_loss)

                # print
                train_loss = running_loss / 2000
                print('[%d, %5d] train loss: %.3f, val loss: %.3f' % (epoch + 1, i + 1, train_loss, val_loss))
                running_loss = 0.0

    # save model
    print('finished training, saving the model...')
    torch.save(net.state_dict(), 'trained_net.pt')


if __name__ == "__main__":
    train()
