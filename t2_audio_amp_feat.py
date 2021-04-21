import pickle
import os
import numpy as np
# import librosa


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


def feat_extract(audio_addr, save_name=None):
    data = {}
    file_pth = audio_addr  # audio_addr只能为单个文件的路径
    fp = open(file_pth, 'rb')
    file_data = pickle.load(fp)
    fp.close()
    audio_all = file_data['audio']  # 4通道的2维矩阵数据，通道i的数据为audio_all[:,i]
    # 预处理：对每一点求功率（平方）
    square = np.power(audio_all, 2)
    max_column = np.amax(square, axis=0)  # 每一列的最大值
    index = np.argmax(square, axis=0)  # 每列最大值的行序号
    maxium = max(max_column)
    col_index = np.argmax(max_column)
    row_index = index[col_index]
    file_name = file_pth.split("/")[-1]
    data[file_name] = [maxium, col_index+1]
    # print("index=%d, p=%f" %(col_index+1, maxium))
    '''
    threshold=0.9 * max(map(max,audio_all))  #自适应阈值
    temp=np.zeros((1,4))
    delta_t=[]
    delta_A=[]
    for timepoint in range(audio_all.shape[0]):
        flag=0  #碰撞点标志
        if timepoint==0:
            continue
        else:  #若当前时刻幅度超过前一时刻幅度乘以阈值，则认定为碰撞时刻
            for i in range(0,4): 
                if audio_all[timepoint,i] >= threshold * audio_all[timepoint-1,i]:
                    flag=1
                    temp[0,i]=audio_all[timepoint,i] #记录幅度值
            if flag==1:
                delta_t.append(timepoint)  #记录碰撞时刻
                delta_A.append(np.argmax(temp)+1)  #记录碰撞的边
    #print(threshold)
    for i in range(len(delta_t)):
        print ("t=%f, A=%f" %(delta_t[i],delta_A[i]))
    '''
    # print(data)
    if not(save_name is None):
        np.save(save_name, data)
    return data


def audio_process(audio_addr):
    '''输入task2或task3一个测试组中一个音频的特征，提取出该音频的特征

    Args:
        audio_addr : 例：'.dataset/task2/test/0/audio_0000.pkl'
            audio_addr只能是一个文件的路径
    
    Returns:
        data: {'音频名'，[音频文件提取出的2个特征]}
    '''
    # audio_addr = get_audio_pth(set_dir)
    data = feat_extract(audio_addr)
    return data


def main():
    set_dir = './dataset/task2/test/0/audio_0000.pkl'
    # save_name = 't2_0_audio_feats.npy'
    audio_process(set_dir)


if __name__ == '__main__':  # 调用此文件时先从此处执行
    main()
