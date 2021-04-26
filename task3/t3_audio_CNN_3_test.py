import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import t2_audio_process

np.set_printoptions(threshold=np.inf)


class AudioDataset_test(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = np.load(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = np.reshape(self.data[idx, 0:], (4, 40, 1))
        data = torch.tensor(data)
        data = data.type(torch.FloatTensor)

        return data


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


def test_rcnet(model, test_iter, device):
    result = []
    model.eval()
    for X in test_iter:
        X = X.to(device)
        output = model(X)
        _, preds = torch.max(output, 1)
        obj = preds[0].item()
        result.append(obj)
    return result


def set_audio_classify(set_dir):
    '''对task2每组测试数据的音频进行分类，返回字典形式的分类结果

    Args:
        set_dir (字符串): 例：'.dataset/task2/test/0'

    Returns:
        [字典]: 例：{audio_0000: 2, audio_0001: 3}
    '''
    batch_size = 1
    # load model
    model = torch.load('t1_cnn.pth')  # 使用相同的网络参数

    ###########################################################################
    set_num = set_dir[-1]
    data_name = 't3_'+set_num+'_audio_feats.npy'
    t2_audio_process.audio_process(set_dir, data_name)  # 计算mfcc结果
    ###########################################################################

    test_dataset = AudioDataset_test(data_name)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_list = test_rcnet(model, test_dataloader, device)  # 返回所有判断结果的list
    # 制成字典
    result_dict = {}
    counter = 0
    for file_name in os.listdir(set_dir):
        file_pth = set_dir + '/' + file_name
        if not os.path.isdir(file_pth):  # 各音频文件的名字
            result_dict.update({file_name: result_list[counter]})
        counter += 1
    return result_dict


def main():
    set_dir = './dataset/task2/test/0'
    result_dict = set_audio_classify(set_dir)
    return result_dict


if __name__ == '__main__':
    main()
