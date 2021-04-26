import cv2
import os
import numpy as np
from math import sqrt
import pickle


def read_mask(root_path):
    video = {}
    for file_path in os.listdir(root_path):
        if os.path.isdir(root_path + '/' + file_path):
            image_path = root_path + '/' + file_path + '/' + 'mask'
            img_msk = []
            for image_path2 in os.listdir(image_path):
                full_path = image_path + '/' + image_path2
                temp1 = cv2.imread(full_path, flags=2)
                temp1 = np.array(temp1)
                img_msk.append(temp1)
            img_msk = np.array(img_msk)
            video[file_path] = img_msk
    return video


def extract_pos(img):
    global temp_x
    contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_size = 0
    for i in range(len(contours)):
        if len(contours[i]) > max(max_size, 10):
            M = cv2.moments(contours[i])
            temp_x = int(M["m10"] / M["m00"])
            temp_y = int(M["m01"] / M["m00"])
            max_size = len(contours[i])
    center_x = temp_x
    center_y = temp_y
    return center_y, center_x


def collision_feats(y):
    delta = []
    for i in range(len(y)):
        if i > 0:
            temp = [y[i, 0] - y[i - 1, 0], y[i, 1] - y[i - 1, 1]]
            delta.append(temp)
    delta = np.array(delta)
    delta = delta.reshape([-1, 2])
    delta_len = np.sum(delta ** 2, axis=1)
    feats_1 = np.amax(delta_len)
    landmarks = np.argmax(delta_len)
    if landmarks == len(delta_len)-1:
        delta_len[landmarks] = 0
        temp = feats_1
        feats_1 = np.amax(delta_len)
        landmarks = np.argmax(delta_len)
        delta_len[landmarks] = temp
    if feats_1 > 200:
        for j in range(landmarks + 1, len(delta_len) + 1):
            if j == len(delta_len):
                distance_y = 440 - y[landmarks, 0]
                distance_x = 440 - y[landmarks, 1]
                judge = [distance_y, y[landmarks, 1], y[landmarks, 0], distance_x]
                edge = judge.index(min(judge)) + 1
                belief = 0
                return feats_1, [edge, belief]
            else:
                if delta_len[j] < 50:
                    belief = 1
                    if delta[j - 1, 0] == 0:
                        if delta[j - 1, 1] > 0:
                            edge = 2
                            return feats_1, [edge, belief]
                        else:
                            edge = 4
                            return feats_1, [edge, belief]
                    elif delta[j - 1, 1] == 0:
                        if delta[j - 1, 0] > 0:
                            edge = 1
                            return feats_1, [edge, belief]
                        else:
                            edge = 3
                            return feats_1, [edge, belief]
                    else:
                        # distance_y = y[j - 1, 0] - 440
                        # distance_x = y[j - 1, 1] - 440
                        # judge = [distance_y / delta[j - 1, 0], y[j - 1, 1] / delta[j - 1, 1],
                        #          distance_x / delta[j - 1, 1],
                        #          y[j - 1, 0] / delta[j - 1, 0]]
                        # judge_2 = [i for i in judge if i > 0]
                        # edge = judge.index(min(judge_2)) + 1
                        distance_y = 440 - y[j, 0]
                        distance_x = 440 - y[j, 1]
                        judge = [distance_y, y[j, 1], y[j, 0], distance_x]
                        edge = judge.index(min(judge)) + 1
                        return feats_1, [edge, belief]
                elif np.sum(delta[j - 1, :] * delta[j, :]) / sqrt(delta_len[j]) / sqrt(delta_len[j - 1]) > 0.9:
                    continue
                else:
                    belief = 1
                    if delta[j - 1, 0] == 0:
                        if delta[j - 1, 1] > 0:
                            edge = 4
                            return feats_1, [edge, belief]
                        else:
                            edge = 2
                            return feats_1, [edge, belief]
                    elif delta[j - 1, 1] == 0:
                        if delta[j - 1, 0] > 0:
                            edge = 3
                            return feats_1, [edge, belief]
                        else:
                            edge = 1
                            return feats_1, [edge, belief]
                    else:
                        distance_y = 440 - y[j - 1, 0]
                        distance_x = 440 - y[j - 1, 1]
                        judge = [distance_y / delta[j - 1, 0], -y[j - 1, 1] / delta[j - 1, 1],
                                 -y[j - 1, 0] / delta[j - 1, 0],
                                 distance_x / delta[j - 1, 1]]
                        judge_2 = [i for i in judge if i > 0]
                        edge = judge.index(min(judge_2)) + 1
                        # distance_y = y[j-1, 0] - 440
                        # distance_x = y[j-1, 1] - 440
                        # judge = [distance_y / delta[j-1, 0], y[j-1, 1] / delta[j-1, 1],
                        #          distance_x / delta[j-1, 1], y[j-1, 0] / delta[j-1, 0]]
                        # judge_2 = [i for i in judge if i > 0]
                        # edge = judge.index(min(judge_2)) + 1
                        return feats_1, [edge, belief]
    else:
        distance_y = 440 - y[landmarks, 0]
        distance_x = 440 - y[landmarks, 1]
        judge = [distance_y, y[landmarks, 1], y[landmarks, 0], distance_x]
        edge = judge.index(min(judge)) + 1
        belief = 0
        return feats_1, [edge, belief]


def get_image_feature(root_path):
    img = read_mask(root_path)
    # y = []
    # case = []
    # for i in range(len(img['video_0006'])):
    #     image_temp = img['video_0006'][i,:,:]
    #     x = extract_pos(image_temp)
    #     y.append(x)
    # y = np.array(y)
    # case = collision_feats(y)
    # case = np.array(case)
    img_dict = {}
    for name in os.listdir(root_path):
        y = []
        case = []
        if os.path.isdir(root_path + '/' + name):
            for i in range(len(img[name])):
                image_temp = img[name][i, :, :]
                x = extract_pos(image_temp)
                y.append(x)
            y = np.array(y)
            case = collision_feats(y)
            img_dict[name] = case
    with open("t3_im_feature.pkl", 'wb') as f:  # 写文件
        pickle.dump(img_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    root_path = './dataset/task3/test/0'
    get_image_feature(root_path)
