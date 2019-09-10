import os.path
import random
import glob
import math
from .listdatasets import ListDataset_random_mid, AIM_Challenge_loader
import numpy

def make_dataset(root='', split=0.0, train_or_test='train'):
    """
        60fps

        Our dataset has the following structure.
        Each folder have 181 frames.
        We use 9 frams and a clip.
        For example:

        clip_0 : 0 2 4 6 8 10 12 14 16
        clip_1 : 18 ................34
           |
        clip_19: .....................

        #------------------------------------------------------#

    Returns: framesPath

    """

    Ban_list = []

    dataset_path = ''
    if train_or_test == 'train':
        dataset_path = os.path.join(root, "train/train_60fps")
    elif train_or_test == 'test':
        dataset_path = os.path.join(root, "val/val_60fps")
    else:
        print("Error")

    # load file list
    framesPath = []

    # Find and loop over all the frames in the dir
    for index, folder in enumerate(sorted(os.listdir(dataset_path))):  # folder 0 1 2 3 4 5 ...

        if folder in Ban_list:
            print("ban dir: ", folder)
            continue

        folder_path = os.path.join(dataset_path, folder)
        frames_list = sorted(os.listdir(folder_path))  # frames 0 1 2 3 4 5 ...
        frames_len = len(frames_list)

        first_clips_index = [0, 2, 4, 6, 8, 10, 12, 14, 16]

        step = 18
        clips_num = int(numpy.floor(frames_len / 9))

        for i in range(clips_num):

            if i == 0:
                clips_index = first_clips_index
            else:
                clips_index = [j + step for j in clips_index]

            temp_paths = []

            for clip_index in clips_index:
                temp_name = str(clip_index).zfill(8) + '.png'
                temp_path = os.path.join(folder_path, temp_name)
                temp_paths.append(temp_path)

            framesPath.append(temp_paths)

    random.shuffle(framesPath)

    split_index = int(math.floor(len(framesPath) * split / 100.0))
    assert (split_index >= 0 and split_index <= len(framesPath))

    return (framesPath[:split_index], framesPath[split_index:]) if split_index < len(framesPath) else (framesPath, [])


# use 1% of the samples to be a validation dataset
def AIM_Challenge(root, split=1.0, single='', task='interp', middle=False, high_fps=False):
    if middle == True:
        print("just select the middle frame")
    else:
        print("random select the middle frame")

    train_list, test_list = make_dataset(root=root, train_or_test='train', split=split)
    # test_list = make_dataset(root=root,train_or_test='test')

    train_dataset = ListDataset_random_mid(root, train_list, loader=AIM_Challenge_loader, middle=middle,
                                           high_fps=high_fps)
    test_dataset = ListDataset_random_mid(root, test_list, loader=AIM_Challenge_loader, middle=middle,
                                          high_fps=high_fps)

    return train_dataset, test_dataset
