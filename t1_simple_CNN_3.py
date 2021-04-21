import torch
from torch.utils.data import Dataset, DataLoader
import time
import copy
from torch import nn
import numpy as np
import os
import t1_audio_process


class AudioDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = np.load(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = np.reshape(self.data[idx, 1:], (4, 40, 1))
        landmarks = torch.tensor(self.data[idx, 0])
        data = torch.tensor(data)
        data = data.type(torch.FloatTensor)

        return data, landmarks


class RCNet(nn.Module):
    def __init__(self, num_classes=10):
        super(RCNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 5 * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 5 * 4)
        x = self.classifier(x)
        return x


def train_rcnet(net,
                train_iter,
                val_iter,
                optimizer,
                lr_scheduler,
                device,
                num_epochs=10):
    since = time.time()
    net = net.to(device)
    print("Training on", device, "for", num_epochs, "epochs")
    loss = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(net.state_dict())
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time(
        )
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)  # 送入神经网络
            l = loss(y_hat, y.long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        val_acc = evaluate_accuracy(val_iter, net)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(net.state_dict())
        print(
            'epoch %d, loss %.4f, train acc %.4f, val acc %.4f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               val_acc, time.time() - start))
        lr_scheduler.step()

    time_elapsed = time.time() - since  # 计算用时
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    net.load_state_dict(best_model_wts)
    return net


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(
                    dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(
                        dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def main():
    batch_size = 16

    data_name = 't1_train_feats.npy'
    if not os.path.exists(data_name):
        t1_audio_process.train()  # 若没有mfcc结果，则再调用函数

    dataset = AudioDataset(data_name)

    train_db, val_db = torch.utils.data.random_split(dataset, [
        round(len(dataset) * 0.8),
        (len(dataset) - round(len(dataset) * 0.8))
        ])

    audio_datasets = {'train': train_db, 'val': val_db}

    dataloaders = {x: DataLoader(audio_datasets[x],
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4)
                                 for x in ['train', 'val']}


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = RCNet(len(dataset))

    lr, num_epochs = 0.001, 800
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20], gamma=0.1)

    net = train_rcnet(net, dataloaders['train'], dataloaders['val'], optimizer,
                      lr_scheduler, device, num_epochs)
    torch.save(net, 't1_cnn.pth')  # 保存整个模型


if __name__ == '__main__':
    main()

