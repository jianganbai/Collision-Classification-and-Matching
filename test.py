from all_net_def import *
import sys
sys.path.append('task1/')
sys.path.append('task2/')
sys.path.append('task3/')
import t1_simple_CNN_3_test as t1_test
import t2_top
import t3_top
import time


def test_task1(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task1/test/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 1, ‘audio_0001’: 3, ...}
    class number:
        ‘061_foam_brick’: 0
        'green_basketball': 1
        'salt_cylinder': 2
        'shiny_toy_gun': 3
        'stanley_screwdriver': 4
        'strawberry': 5
        'toothpaste_box': 6
        'toy_elephant': 7
        'whiteboard_spray': 8
        'yellow_block': 9
    '''
    results = t1_test.main(root_path)
    return results


def test_task2(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task2/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 23, ‘audio_0001’: 11, ...}
    This means audio 'audio_0000.pkl' is matched to video 'video_0023' and ‘audio_0001’ is matched to 'video_0011'.
    '''
    results = t2_top.set_matching(root_path)
    return results


def test_task3(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task3/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': -1, ‘audio_0001’: 12, ...}
    This means audio 'audio_0000.pkl' is not matched to any video and ‘audio_0001’ is matched to 'video_0012'.
    '''
    results = t3_top.set_matching(root_path)
    return results


if __name__ == '__main__':
    '''
    # task1
    time_start = time.time()
    root_path = './dataset/task1/test/'
    result = test_task1(root_path)

    time_end = time.time()
    gap = time_end-time_start
    minutes = int(gap/60)
    seconds = int(gap - 60*minutes)
    print('task1: '+str(minutes)+'min '+str(seconds)+'sec')
    # task2 or taks3
    '''
    for i in range(0, 10):
        time_start = time.time()
        root_path = './dataset/task2/test/'+str(i)+'/'
        result = test_task2(root_path)

        time_end = time.time()
        gap = time_end-time_start
        minutes = int(gap/60)
        seconds = int(gap - 60*minutes)
        print('set'+str(i)+': '+str(minutes)+'min '+str(seconds)+'sec')

    input()
