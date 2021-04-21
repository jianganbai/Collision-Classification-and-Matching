import numpy as np
import copy


def loss_func(sf0, sf1, bf0, bf1, confi):
    # 权重越大越重要，基础权重要将2种特征的影响力调到等额
    weight0_base = 3  # 强度的基础权重，差距很小则权重有奖励，强度差得越大则权重有惩罚
    weight1_base = 1  # 碰撞边的基础权重，置信系数=0则权重很小

    # 特征0的损失函数
    # 第1列为差值区间，第2列为附加权重
    feat0_loss_ref = np.array([[0, 0.05, 0.1, 0.7, 0.9, 1], [0.4, 0.7, 1, 2, 5, 100]])
    d0 = abs(sf0-bf0)
    for i in range(1, feat0_loss_ref.shape[1]):
        start = feat0_loss_ref[0, i-1]
        end = feat0_loss_ref[0, i]
        if (start <= d0) and (d0 <= end):
            loss0 = d0*weight0_base*feat0_loss_ref[1, i-1]
            break

    # 特征1的损失函数
    unknown_weight = 0.2
    d1 = abs(sf1-bf1) % 2  # 距离至多差2
    loss1 = d1*weight1_base*max(confi, unknown_weight)

    loss = loss0 + loss1
    return loss


def traverse(small_mat, big_mat, hit_confi, small_type, small_num, level,
             num_remain, current_match, current_loss, loss_min, best_match):
    # audio没有可匹配的或video没有可匹配的
    if (level >= small_num) or (len(num_remain) == 0):
        if current_loss < loss_min:
            best_match = copy.deepcopy(current_match)
            loss_min = current_loss
            return best_match, loss_min
    # 还可以再匹配
    else:
        # 使用小集合去匹配大集合
        for big_idx in num_remain:
            sf0 = small_mat[level, 0]
            sf1 = small_mat[level, 1]
            bf0 = big_mat[big_idx, 0]
            bf1 = big_mat[big_idx, 1]
            if small_type == 'audio':  # 视频是大集合
                confi = hit_confi[big_idx]
            else:  # 等大或者音频是大集合
                confi = hit_confi[level]
            addition_loss = loss_func(sf0, sf1, bf0, bf1, confi)
            loss_new = current_loss + addition_loss
            if loss_new >= loss_min:
                continue
            else:  # 这一步还没有超过loss_min
                level_new = level+1
                num_remain_new = copy.deepcopy(num_remain)
                num_remain_new.remove(big_idx)
                match_new = copy.deepcopy(current_match)
                match_new[level] = big_idx
                # 递归
                best_match, loss_min = traverse(small_mat, big_mat,
                                                hit_confi, small_type,
                                                small_num, level_new,
                                                num_remain_new, match_new,
                                                loss_new, loss_min,
                                                best_match)
        return best_match, loss_min


def set_within_class_match(video_feat_dict, audio_feat_dict):
    '''针对一个测试组中的一类物体间的匹配，已提取出2维特征
        第1维特征为（视频最大帧间位移，音频最大幅度）
        第2维特征为（视频预测碰撞边，音频预测碰撞边）

    Args:
        video_feat_dict (字典): {‘视频文件夹名’：[最大位移值，[最大速度后碰撞边编号，置信系数]]}
        audio_feat_dict (字典): {‘音频名’：[短时功率的最大值，最明显碰撞的碰撞边编号]}

    Returns:
        match_result : {'音频名': '匹配上的视频文件夹名'}
        unmatched : # {‘video’ or 'audio': [名字]}，没有未匹配上的则返回match
    '''
    max_intense = 1  # 特征0的正则化系数

    # 若有1个为空，则直接返回
    if not bool(video_feat_dict):  # video为空
        if not bool(audio_feat_dict):  # audio为空
            match_result = {}
            unmatched = None
        else:
            match_result = {}
            unmatched = {'audio': list(audio_feat_dict.keys())}
        return match_result, unmatched
    if not bool(audio_feat_dict):  # audio为空
        match_result = {}
        unmatched = {'video': list(video_feat_dict.keys())}
        return match_result, unmatched

    # 转化为矩阵形式，去除文件名
    video_feat_list = []
    hit_confi = []
    for video_feat in video_feat_dict.values():
        video_feat_list.append([video_feat[0], video_feat[1][0]])
        hit_confi.append(video_feat[1][1])  # 碰撞边的置信系数
    audio_feat_list = []
    for audio_feat in audio_feat_dict.values():
        audio_feat_list.append(audio_feat)
    video_feat_mat = np.array(video_feat_list, dtype='float64')
    audio_feat_mat = np.array(audio_feat_list, dtype='float64')
    # 对第1维特征做归max_intense化
    video_1_max = max(video_feat_mat[:, 0].max(), 1)
    video_feat_mat[:, 0] = video_feat_mat[:, 0]/video_1_max*max_intense
    audio_1_max = max(audio_feat_mat[:, 0].max(), 1)
    audio_feat_mat[:, 0] = audio_feat_mat[:, 0]/audio_1_max*max_intense

    # 找到样本数小的集合和样本数大的集合
    video_num = video_feat_mat.shape[0]
    audio_num = audio_feat_mat.shape[0]
    if video_num > audio_num:  # 音频是小集合
        small_mat = audio_feat_mat
        big_mat = video_feat_mat
        small_type = 'audio'
    else:  # 等大或视频是小集合
        small_mat = video_feat_mat
        big_mat = audio_feat_mat
        small_type = 'equal_or_video'

    # 使用递归进行先序遍历
    small_num = small_mat.shape[0]
    big_num = big_mat.shape[0]
    num_remain = list(range(0, big_num))
    current_match = -1*np.ones(small_num, dtype='int')
    loss_min = 10000
    best_match, _ = traverse(small_mat=small_mat, big_mat=big_mat,
                             hit_confi=hit_confi, small_type=small_type,
                             small_num=small_num, level=0,
                             num_remain=num_remain, current_match=current_match,
                             current_loss=0, loss_min=loss_min,
                             best_match=current_match)

    # 制成字典
    all_video_name = list(video_feat_dict.keys())
    all_audio_name = list(audio_feat_dict.keys())
    match_result = {}  # {'audio_0001': 'video_0034'}
    unmatched = None  # {‘video’ or 'audio': [名字]}
    if video_num > audio_num:  # 视频数>音频数
        # 整合所有匹配结果
        for audio_counter, audio_name in enumerate(all_audio_name):
            video_idx = best_match[audio_counter]
            video_name = all_video_name[video_idx]  # 匹配上的视频的文件夹名
            single_match = {audio_name: video_name}
            match_result.update(single_match)
        # 整合所有未匹配结果
        unmatched_video = []
        for i in range(0, len(video_feat_dict)):
            if not (i in best_match):  # 视频没有匹配上
                video_name = all_video_name[i]
                unmatched_video.append(video_name)
        if len(unmatched_video) > 0:
            unmatched = {'video': unmatched_video}
    else:  # 视频数<=音频数
        # 整合所有匹配结果为audio的索引
        for video_counter, video_name in enumerate(all_video_name):
            audio_name = all_audio_name[best_match[video_counter]]
            single_match = {audio_name: video_name}
            match_result.update(single_match)
        unmatched_audio = []
        for i in range(0, len(audio_feat_dict)):
            if not (i in best_match):  # 音频没有匹配上
                audio_name = all_audio_name[i]
                unmatched_audio.append(audio_name)
        if len(unmatched_audio) > 0:
            unmatched = {'audio': unmatched_audio}
    return match_result, unmatched


def main():
    video_feats = {'video00': [0, [1, 1]], 'video01': [1, [0, 1]], 'video2': [0.7, [1, 1]], 'video3': [0.8, [1, 1]]}
    audio_feats = {'audio0': [1, 0], 'audio1': [0.8, 1]}
    set_within_class_match(video_feats, audio_feats)
    input()


if __name__ == '__main__':
    main()
