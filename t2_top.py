import os
import pickle
# 第1步分类匹配相关
import t2_im_CNN_1_test as im_CNN
import t2_audio_CNN_3_test as audio_CNN
from all_net_def import *
# 第2步特征匹配相关
import t2_set_within_class_match as class_match
import t2_im_edge_feat as im_2feats
import t2_audio_amp_feat as audio_2feats


def first_classify(set_dir):
    '''使用神经网络分别对视频和音频进行分类

    Args:
        set_dir : 每个测试组的地址

    Returns:
        mix_class_distri : 按照分类结果排列的字典，如{0: [[video], [audio]]}
    '''
    
    # 图像分类
    im_class_dict = im_CNN.set_im_classify(set_dir)
    # 音频分类
    audio_class_dict = audio_CNN.set_audio_classify(set_dir)
    '''
    # 保存字典，方便调试
    f1 = open('tt2_im_dict_temp', 'w')
    f1.write(str(im_class_dict))
    f1.close()
    f2 = open('tt2_audio_dict_temp', 'w')
    f2.write(str(audio_class_dict))
    f2.close()
    
    f1 = open('tt2_im_dict_temp', 'r')
    im_temp = f1.read()
    im_class_dict = eval(im_temp)
    f1.close()
    f2 = open('tt2_audio_dict_temp', 'r')
    audio_temp = f2.read()
    audio_class_dict = eval(audio_temp)
    f2.close()
    '''
    # 统计图像分类结果
    im_class_distri = {}
    for i in range(0, 10):  # 初始化字典
        im_class_distri.update({i: []})
    for name, im_class in im_class_dict.items():
        im_class_distri[im_class].append(name)
    # 统计音频分类结果
    audio_class_distri = {}
    for i in range(0, 10):
        audio_class_distri.update({i: []})
    for name, audio_class in audio_class_dict.items():
        audio_class_distri[audio_class].append(name)
    # 整合图像分类结果和音频分类结果
    mix_class_distri = {}
    for i in range(0, 10):
        mix_class_distri.update({i: []})
    for class_num in audio_class_distri:
        mix_class_distri[class_num].append(im_class_distri[class_num])
        mix_class_distri[class_num].append(audio_class_distri[class_num])
    return mix_class_distri


def second_classify(set_dir, mix_class_distri):
    '''视频和音频分别提取出2类特征：最大速度+最大幅度，碰撞边+碰撞边

    Args:
        set_dir : 每个测试组的地址
        mix_class_distri : {类编号: [[所有视频名], [所有音频名]]}

    Returns:
        match_comb : {'音频名': '视频名'}
    '''
    # 视频特征提取接口
    im_2feats.get_image_feature(set_dir)  # 分析测试组所有文件，保存为pkl文件
    fp = open('t2_im_feature.pkl', 'rb')
    all_video_2feats_dict = pickle.load(fp)  # 注意提出来的数据是np，不是列表
    fp.close()

    match_comb = {}  # 未按照音频升序排列的字典
    unmatched_video = []
    unmatched_audio = []
    for obj_group in mix_class_distri.values():
        video_group = obj_group[0]  # 该类别内所有的视频名
        audio_group = obj_group[1]  # 该类别内所有的音频名

        # 提取同一类物体视频特征
        video_feat_dict = {}
        for video_name in video_group:
            video_2feats = all_video_2feats_dict[video_name]
            video_feat_dict.update({video_name: video_2feats})

        # 提取同一类物体音频特征
        audio_feat_dict = {}
        for audio_name in audio_group:
            audio_pth = set_dir+'/'+audio_name
            audio_feat = audio_2feats.feat_extract(audio_pth)
            audio_feat_dict.update(audio_feat)

        # 加权全匹配
        class_matched, unmatched = class_match.set_within_class_match(video_feat_dict, audio_feat_dict)
        # 已匹配上的直接加入字典
        match_comb.update(class_matched)
        # 未匹配上的先按'video'or'audio'分别合并在一起
        if not(unmatched is None):
            file_type = list(unmatched.keys())[0]  # 未匹配上的类型
            if file_type == 'video':
                for i in range(len(unmatched['video'])):
                    unmatched_video.append(unmatched['video'][i])
            else:
                for i in range(len(unmatched['audio'])):
                    unmatched_audio.append(unmatched['audio'][i])

    # 所有未匹配的再进行一次匹配
    unmatched_video_dict = {}
    unmatched_audio_dict = {}
    for i in range(0, len(unmatched_video)):
        unmatched_video_dict.update({unmatched_video[i]:
                                     all_video_2feats_dict[unmatched_video[i]]})
    for i in range(0, len(unmatched_audio)):
        audio_pth = set_dir+'/'+unmatched_audio[i]
        audio_feat = audio_2feats.feat_extract(audio_pth)
        unmatched_audio_dict.update(audio_feat)
    class_matched, _ = class_match.set_within_class_match(unmatched_video_dict,
                                                          unmatched_audio_dict)

    # 再次整合
    match_comb.update(class_matched)
    return match_comb


def match_ascend_seq(set_dir, match_raw):
    # 看有多少组（视频，音频）
    all_audio_name = []
    for file_name in os.listdir(set_dir):
        file_pth = set_dir+'/'+file_name
        if not (os.path.isdir(file_pth)):
            all_audio_name.append(file_name)
    # 按照音频编号递增顺序重新生成字典，同时简写视频名
    match_ascend = {}
    for audio_name in all_audio_name:
        video_name = match_raw[audio_name]
        if video_name is None:
            video_idx = -1
        else:
            video_idx = int(video_name[-2]+video_name[-1])
        match_ascend.update({audio_name: video_idx})
    return match_ascend


def set_matching(root_dir):
    set_dir = root_dir[0: len(root_dir)-1]  # 去掉最后的/
    mix_class_distri = first_classify(set_dir)
    match_raw = second_classify(set_dir, mix_class_distri)
    match_ascend = match_ascend_seq(set_dir, match_raw)
    return match_ascend


if __name__ == '__main__':
    root_dir = './dataset/task2/mytest/1/'
    set_matching(root_dir)
