from all_net_def import *
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import t1_audio_process

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


def main(test_pth='./dataset/task1/test'):
    batch_size = 1
    # load model
    model = torch.load('t1_cnn.pth')

    data_name = 't1_test_feats.npy'
    t1_audio_process.test()

    test_dataset = AudioDataset_test(data_name)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_list = test_rcnet(model, test_dataloader, device)  # 返回所有判断结果的list
    # 制成字典
    result_dict = {}
    counter = 0
    for file_name in os.listdir(test_pth):  # 测试集各音频文件名
        result_dict.update({file_name: result_list[counter]})
        counter += 1
    return result_dict


if __name__ == '__main__':
    main()
