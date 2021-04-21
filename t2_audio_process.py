import pickle
import os
import numpy as np
import librosa


def get_audio_pth(dir_pth):
    '''针对task2或task3的一个测试组，返回其中各个音频的地址

    Args:
        dir_pth: 例:'./dataset/task2/test/0'

    Returns:
        audio_addr: 该测试组内各个音频的地址
    '''
    audio_addr = []
    for file_name in os.listdir(dir_pth):
        file_pth = dir_pth + '/' + file_name
        if not os.path.isdir(file_pth):  # 剔除所有文件夹
            audio_addr.append(file_pth)
    return audio_addr


def feat_extract(audio_addr, save_name):
    data = None
    for file_pth in audio_addr:
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
        # file_feat存的是一个音频文件4个通道的全部特征
        if data is None:
            data = file_feat
        else:
            data = np.append(data, file_feat, axis=0)
    # data按行遍历，从上至下分别为各个音频文件的特征
    np.save(save_name, data)


def audio_process(set_dir, save_name):
    '''输入task2或task3一个测试组的地址，提取出其中各个音频的特征

    Args:
        dir_pth : 例:'./dataset/task2/test'
        save_name : 保存数据文件的名字
    '''
    audio_addr = get_audio_pth(set_dir)
    feat_extract(audio_addr, save_name)


def main():
    set_dir = './dataset/task2/test/0'
    save_name = 't2_0_audio_feats.npy'
    audio_process(set_dir, save_name)


if __name__ == '__main__':  # 调用此文件时先从此处执行
    main()
