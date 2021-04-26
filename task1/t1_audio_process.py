import pickle
import os
import numpy as np
import librosa
# import pandas as pd


def get_train_name():
    '''
        返回各类别音频的字典，格式为{<类名>，<该类音频地址的列表>}
    '''
    train_pth = './dataset/train'
    train_dict = {}
    for class_name in os.listdir(train_pth):  # 标签名
        class_pth = train_pth+'/'+class_name
        class_file = []
        for file_name in os.listdir(class_pth):
            class_file.append(class_pth+'/'+file_name)
        dict_temp = {class_name: class_file}
        train_dict.update(dict_temp)
    return train_dict


def get_test_name():
    '''
        返回各类别音频地址的list
    '''
    test_pth = './dataset/task1/test'
    test_addr = []
    for file_name in os.listdir(test_pth):  # 测试集各音频文件名
        test_addr.append(test_pth+'/'+file_name)
    return test_addr


def read_train_audio(train_dict):
    data = None
    index = 0
    for class_name, file_all_pth in train_dict.items():  # class_name是类名
        for file_pth in file_all_pth:
            full_pth = file_pth+'/'+'audio_data.pkl'
            fp = open(full_pth, 'rb')
            file_data = pickle.load(fp)
            fp.close()
            audio_all = file_data['audio']  # 4通道的2维矩阵数据，通道i的数据为audio_all[:,i]
            # 在此添加提取特征函数
            file_feat = np.array([index])
            for i in range(0, 4):
                mfccs = librosa.feature.mfcc(audio_all[:, i], sr=44100, n_mfcc=40)
                # 各类特征值
                mfcc_mean = np.mean(mfccs, axis=1)
                '''
                mfcc_std = np.std(mfccs, axis=1)
                mfcc_skew = scipy.stats.skew(mfccs, axis=1)
                mfcc_d1 = librosa.feature.delta(mfccs)
                mfcc_d1_mean = np.mean(np.power(mfcc_d1, 2), axis=1)
                '''
                one_channel_data = np.hstack((mfcc_mean))  # 保留其它特征的接口

                file_feat = np.hstack((file_feat, one_channel_data))
            file_feat = file_feat.reshape(1, len(file_feat))  # 1维数组转2维数组
            if data is None:
                data = file_feat
            else:
                data = np.append(data, file_feat, axis=0)

        index = index+1
    np.save('t1_train_feats', data)


def read_test_audio(test_addr):
    data = None
    for file_pth in test_addr:
        fp = open(file_pth, 'rb')
        file_data = pickle.load(fp)
        fp.close()
        audio_all = file_data['audio']  # 4通道的2维矩阵数据，通道i的数据为audio_all[:,i]
        # 在此添加提取特征函数
        file_feat = np.array([])
        for i in range(0, 4):
            mfccs = librosa.feature.mfcc(audio_all[:, i], sr=44100, n_mfcc=40)
            # 各类特征值
            mfcc_mean = np.mean(mfccs, axis=1)
            '''
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_skew = scipy.stats.skew(mfccs, axis=1)
            mfcc_d1 = librosa.feature.delta(mfccs)
            mfcc_d1_mean = np.mean(np.power(mfcc_d1, 2), axis=1)
            '''
            one_channel_data = np.hstack((mfcc_mean))
            file_feat = np.hstack((file_feat, one_channel_data))
        file_feat = file_feat.reshape(1, len(file_feat))  # 1维数组转2维数组
        if data is None:
            data = file_feat
        else:
            data = np.append(data, file_feat, axis=0)
    np.save('t1_test_feats', data)


def train():
    train_dict = get_train_name()
    read_train_audio(train_dict)


def test():
    test_addr = get_test_name()
    read_test_audio(test_addr)


def main(audio_type):
    if audio_type == 'train':
        train()
    elif audio_type == 'test':
        test()
    else:
        print('invalid type')


if __name__ == '__main__':  # 调用此文件时先从此处执行
    audio_type = 'test'
    main(audio_type)
