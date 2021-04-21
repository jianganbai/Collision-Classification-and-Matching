import copy
import time
import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ImNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 3 * 3, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 3 * 3)
        x = self.classifier(x)
        return x


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


def train_rcnet(net,
                train_iter,
                val_iter,
                optimizer,
                lr_scheduler,
                device,
                num_epochs=50):
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
            y_hat = net(X)
            l = loss(y_hat, y)
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

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    net.load_state_dict(best_model_wts)
    return net


class ImageDataset(Dataset):
    def __init__(self, root_dir, image_type, transform=None):
        index = 0
        addr = []  # 保存各图片的路径及其类型：[[地址，类型编号]]
        for class_name in os.listdir(root_dir):  # 标签名
            class_pth = root_dir+'/'+class_name
            for set_num in os.listdir(class_pth):
                video_pth = class_pth+'/'+set_num+'/'+image_type
                for file_name in os.listdir(video_pth):
                    image_pth = video_pth+'/'+file_name
                    addr.append([image_pth, index])
            index += 1
        self.addr = addr
        self.transform = transform

    def __len__(self):
        return len(self.addr)  # 训练样本个数

    def __getitem__(self, idx):  # idx为每次取样本的编号，貌似是随机的int
        if torch.is_tensor(idx):
            idx = idx.tolist()  # idx转化为list
        image_info = self.addr[idx]
        img_name = image_info[0]
        label = image_info[1]

        image = Image.open(img_name)  # 读入图像

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label


def main():
    root_dir = './dataset/train'
    image_type = 'rgb'
    batch_size = 16
    lr, num_epochs = 0.001, 100

    data_transforms = transforms.Compose([
            transforms.Resize((24, 24)),
            transforms.ToTensor()
        ])

    dataset = ImageDataset(root_dir, image_type, data_transforms)

    train_db, val_db = torch.utils.data.random_split(dataset, [
        round(len(dataset) * 0.8),
        (len(dataset) - round(len(dataset) * 0.8))
        ])

    image_datasets = {'train': train_db, 'val': val_db}

    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4)
                   for x in ['train', 'val']}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = ImNet(10)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20], gamma=0.1)

    net = train_rcnet(net, dataloaders['train'], dataloaders['val'], optimizer,
                      lr_scheduler, device, num_epochs)
    torch.save(net, 't2_im_cnn.pth')


if __name__ == '__main__':
    main()
