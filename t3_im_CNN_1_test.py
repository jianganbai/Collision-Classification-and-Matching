import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

np.set_printoptions(threshold=np.inf)


class ImageDataset_test(Dataset):
    def __init__(self, root_dir, image_type, transform=None):
        # root_dir为./dataset/task2/test/i，i为待匹配的组编号
        addr = []  # 保存各图片的路径及其类型：[[地址，类型编号]]
        for video_name in os.listdir(root_dir):  # 标签名
            video_pth = root_dir+'/'+video_name
            if os.path.isdir(video_pth):
                video_pth = video_pth + '/'+image_type
                for file_name in os.listdir(video_pth):
                    image_pth = video_pth+'/'+file_name
                    addr.append(image_pth)  # 每个测试集的图片地址
        self.addr = addr
        self.transform = transform

    def __len__(self):
        return len(self.addr)  # 训练样本个数

    def __getitem__(self, idx):  # idx为每次取样本的编号，貌似是随机的int
        if torch.is_tensor(idx):
            idx = idx.tolist()  # idx转化为list
        img_pth = self.addr[idx]

        image = Image.open(img_pth)  # 读入图像

        if self.transform:
            image = self.transform(image)
        return image


'''
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
'''


def test_imnet(model, test_iter, device):
    result = []
    model.eval()
    for X in test_iter:
        X = X.to(device)
        output = model(X)
        _, preds = torch.max(output, 1)
        obj = preds[0].item()
        result.append(obj)
    return result


def set_im_classify(set_dir):
    '''返回每组测试数据的50个视频分类结果

    Args:
        set_dir (字符串): 每组测试数据的路径，如'.dataset/taske2/test/i'

    Returns:
        set_im_dict: 该组测试数据的分类结果，如{video_0000: 6, video_0001: 3}
    '''
    image_type = 'rgb'
    batch_size = 1
    # load model
    model = torch.load('t2_im_cnn.pth')

    data_transforms = transforms.Compose([
        transforms.Resize((24, 24)),
        transforms.ToTensor()
    ])

    test_dataset = ImageDataset_test(set_dir, image_type, transform=data_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_result_list = test_imnet(model, test_dataloader, device)  # 返回所有图片判断结果的list

    # 同一个video的合在一起
    counter = 0
    set_im_dict = {}
    for dir_name in os.listdir(set_dir):
        dir_pth = set_dir + '/' + dir_name
        if os.path.isdir(dir_pth):
            video_pth = dir_pth + '/' + image_type
            pic_num = len(os.listdir(video_pth))
            video_result_list = all_result_list[counter:counter+pic_num]  # 该视频的图像分类结果
            # 统计众数
            counts = np.bincount(video_result_list)
            obj_type = np.argmax(counts)  # 该视频分类结果
            # 制成字典
            set_im_dict.update({dir_name: obj_type})
            counter += pic_num
    return set_im_dict


def main():
    test_dir = './dataset/task2/test'
    test_im_dict = []
    for set_num in os.listdir(test_dir):
        set_dir = test_dir+'/'+set_num
        set_im_dict = set_im_classify(set_dir)
        # set_im_dict为该测试数据组内50段视频的分类结果，格式如下{文件夹名：分类结果}
        test_im_dict.append({set_num: set_im_dict})
    return test_im_dict


if __name__ == '__main__':
    test_im_dict = main()
    input()
