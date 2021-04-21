import numpy as np
import copy


def loss_func(af0, af1, vf0, vf1, confi):
    # 权重越大越重要，基础权重要将2种特征的影响力调到等额
    weight0_base = 3  # 强度的基础权重，差距很小则权重有奖励，强度差得越大则权重有惩罚
    weight1_base = 1  # 碰撞边的基础权重，置信系数=0则权重很小

    # 特征0的损失函数
    # 第1列为差值区间，第2列为附加权重
    feat0_loss_ref = np.array([[0, 0.05, 0.1, 0.5, 0.8, 1], [0.4, 0.6, 0.8, 1, 3, 100]])
    d0 = abs(af0-vf0)
    for i in range(1, feat0_loss_ref.shape[1]):
        start = feat0_loss_ref[0, i-1]
        end = feat0_loss_ref[0, i]
        if (start <= d0) and (d0 <= end):
            loss0 = d0*weight0_base*feat0_loss_ref[1, i-1]
            break

    # 特征1的损失函数
    unknown_weight = 0.1
    d1 = abs(af1-vf1) % 2  # 距离至多差2
    loss1 = d1*weight1_base*max(confi, unknown_weight)

    loss = loss0 + loss1
    return loss


def find_best_match(audio_feat_mat, video_feat_mat, video_confi):
    '''使用递归，每次给出一组待匹配的音频与视频，找到最佳匹配并返回

    Args:
        audio_feat_mat : 音频特征矩阵
        video_feat_mat : 视频特征矩阵
        video_confi : 视频碰撞边置信系数

    Returns:
        [字典]: 匹配结果，例：{匹配的音频编号: 匹配的视频编号}
    '''
    if audio_feat_mat.shape[0] == 0:  # 已经没有audio匹配
        if video_feat_mat.shape[0] == 0:  # 同时也没有video匹配
            return {'all_empty': 'null'}
        else:
            return {'no_audio': 'null'}
    if video_feat_mat.shape[0] == 0:  # 已经没有video匹配
        return {'no_video': 'null'}

    # 寻找2个非空集合间损失函数最小的配对
    thres = 3

    loss_min = None
    best_match = None
    audio_num = audio_feat_mat.shape[0]
    video_num = video_feat_mat.shape[0]
    for audio_iter in range(0, audio_num):
        for video_iter in range(0, video_num):
            af0 = audio_feat_mat[audio_iter, 0]
            af1 = audio_feat_mat[audio_iter, 1]
            vf0 = video_feat_mat[video_iter, 0]
            vf1 = video_feat_mat[video_iter, 1]
            confi = video_confi[video_iter]
            loss = loss_func(af0, af1, vf0, vf1, confi)
            if loss_min is None:
                loss_min = loss
                best_match = {audio_iter: video_iter}
            elif loss_min > loss:
                loss_min = loss
                best_match = {audio_iter: video_iter}  # 都是编号
    if loss_min < thres:
        return best_match
    else:
        return {'loss_too_big': 'null'}


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
            all_result = None
        else:
            all_result = {}
            for audio_name in audio_feat_dict.keys():
                all_result.update({audio_name: -1})
        return all_result
    if not bool(audio_feat_dict):  # audio为空
        return None

    # 转化为矩阵形式，去除文件名
    video_feat_list = []
    hit_confidence = []
    for video_feat in video_feat_dict.values():
        video_feat_list.append([video_feat[0], video_feat[1][0]])
        hit_confidence.append(video_feat[1][1])  # 碰撞边的置信系数
    audio_feat_list = []
    for audio_feat in audio_feat_dict.values():
        audio_feat_list.append(audio_feat)
    video_feat_mat = np.array(video_feat_list, dtype='float64')
    audio_feat_mat = np.array(audio_feat_list, dtype='float64')
    # 所有文件名
    all_video_name = list(video_feat_dict.keys())
    all_audio_name = list(audio_feat_dict.keys())
    # 对第1维特征做归max_intense化
    video_1_max = max(video_feat_mat[:, 0].max(), 1)
    video_feat_mat[:, 0] = video_feat_mat[:, 0]/video_1_max*max_intense
    audio_1_max = max(audio_feat_mat[:, 0].max(), 1)
    audio_feat_mat[:, 0] = audio_feat_mat[:, 0]/audio_1_max*max_intense

    # 贪心匹配
    match_result = {}  # 所有已匹配音频视频结果：{音频名：视频名}
    unmatch_result = {}  # 所有未匹配的音频：{音频名：-1}
    audio_feat_elimin = copy.deepcopy(audio_feat_mat)
    video_feat_elimin = copy.deepcopy(video_feat_mat)
    video_confi_elimin = copy.deepcopy(hit_confidence)
    audio_name_elimin = copy.deepcopy(all_audio_name)
    video_name_elimin = copy.deepcopy(all_video_name)
    # 开始匹配
    while True:
        best_match = find_best_match(audio_feat_elimin,
                                     video_feat_elimin, video_confi_elimin)
        match_label = list(best_match.keys())[0]  # 匹配结果标识
        if match_label == 'all_empty':  # 已全部匹配上
            break
        elif match_label == 'no_audio':  # 音频已全部匹配上
            break
        # 视频已全部匹配上或损失函数过大
        elif (match_label == 'no_video') or (match_label == 'loss_too_big'):
            for i in range(0, audio_feat_elimin.shape[0]):
                unmatch_format = {audio_name_elimin[i]: -1}
                unmatch_result.update(unmatch_format)
            break
        else:  # 存在满足一组匹配
            audio_idx = match_label
            video_idx = best_match[match_label]
            match_audio_name = audio_name_elimin[audio_idx]
            match_video_name = video_name_elimin[video_idx]
            match_format = {match_audio_name: match_video_name}
            match_result.update(match_format)
            # 约减集合
            audio_feat_elimin = np.delete(audio_feat_elimin, audio_idx, axis=0)
            video_feat_elimin = np.delete(video_feat_elimin, video_idx, axis=0)
            video_confi_elimin = np.delete(video_confi_elimin, video_idx, axis=0)
            del audio_name_elimin[audio_idx]
            del video_name_elimin[video_idx]

    # 简写视频名
    for audio_name, video_name in match_result.items():
        video_idx = int(video_name[-2]+video_name[-1])
        match_result[audio_name] = video_idx
    # 合并匹配结果和未匹配结果
    all_result = {**match_result, **unmatch_result}
    return all_result


def main():
    video_feats = {'video00': [0, [1, 1]], 'video01': [1, [0, 1]], 'video02': [1, [2, 1]]}
    audio_feats = {'audio0': [2, 0], 'audio1': [1, 1], 'audio2': [0, 1], 'audio3': [2, 1]}
    set_within_class_match(video_feats, audio_feats)


if __name__ == '__main__':
    main()
