import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
from skimage import color
import random



def default_loader_random_mid(root, im_path,input_frame_size = (3, 128, 128), output_frame_size = (3, 128, 128), data_aug = True, middle=False, high_fps=False):
    frame_prefix = im_path.split('\\')[-1]


    if data_aug and random.randint(0, 1):
        path_pre2 = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_X0.png")
        path_pre1 = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_y.png")
        path_mid = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_X1.png")
    else:
        path_pre2 = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_X1.png")
        path_pre1 = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_y.png")
        path_mid = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_X0.png")
    # try:  # we can not fail in reading an image
    #     im_pre2 = imread(path_pre2)  # i failed sometimes
    #     im_pre1 = imread(path_pre1)  # i failed sometimes
    #     im_mid = imread(path_mid)
    #     # im_nxt1 = imread(path_nxt1)
    #     # im_nxt2 = imread(path_nxt2)
    #
    # except:
    #     print("\n Read path_pre fail ", path_pre2, '\n', path_mid)
    # else:
    #     # successful read
    #     break
    im_pre2 = imread(path_pre2)  # i failed sometimes
    im_pre1 = imread(path_pre1)  # i failed sometimes
    im_mid = imread(path_mid)

    h_offset = random.choice(range(150 - input_frame_size[1] - 1))
    w_offset = random.choice(range(150 - input_frame_size[2] - 1))

    im_pre2 = im_pre2[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]  # imresize(im_pre, (128, 424))[:, 20:404, :]
    im_pre1 = im_pre1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]  # imresize(im_pre, (128, 424))[:, 20:404, :]
    im_mid = im_mid[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]

    if data_aug:
        if random.randint(0, 1):
            im_pre2 = np.fliplr(im_pre2)
            im_mid = np.fliplr(im_mid)
            im_pre1 = np.fliplr(im_pre1)
        if random.randint(0, 1):
            im_pre2 = np.flipud(im_pre2)
            im_mid = np.flipud(im_mid)
            im_pre1 = np.flipud(im_pre1)

    X0 = np.transpose(im_pre2,(2,0,1))
    X2 = np.transpose(im_mid, (2, 0, 1))


    y = np.transpose(im_pre1, (2, 0, 1))
    return X0.astype("float32")/ 255.0, \
            X2.astype("float32")/ 255.0,\
            y.astype("float32")/ 255.0, 0


def default_loader(root, im_path,input_frame_size = (3, 128, 128), output_frame_size = (3, 128, 128), data_aug = True):
    frame_prefix = im_path.split('\\')[-1]

    if data_aug and random.randint(0, 1):
        path_pre2 = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_X0.png")
        path_pre1 = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_y.png")
        path_mid = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_X1.png")
    else:
        path_pre2 = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_X1.png")
        path_pre1 = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_y.png")
        path_mid = os.path.join(root, '\\'.join(im_path.split('\\')[:-1]), frame_prefix + "_X0.png")
    # try:  # we can not fail in reading an image
    #     im_pre2 = imread(path_pre2)  # i failed sometimes
    #     im_pre1 = imread(path_pre1)  # i failed sometimes
    #     im_mid = imread(path_mid)
    #     # im_nxt1 = imread(path_nxt1)
    #     # im_nxt2 = imread(path_nxt2)
    #
    # except:
    #     print("\n Read path_pre fail ", path_pre2, '\n', path_mid)
    # else:
    #     # successful read
    #     break
    im_pre2 = imread(path_pre2)  # i failed sometimes
    im_pre1 = imread(path_pre1)  # i failed sometimes
    im_mid = imread(path_mid)

    h_offset = random.choice(range(150 - input_frame_size[1] - 1))
    w_offset = random.choice(range(150 - input_frame_size[2] - 1))

    im_pre2 = im_pre2[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]  # imresize(im_pre, (128, 424))[:, 20:404, :]
    im_pre1 = im_pre1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]  # imresize(im_pre, (128, 424))[:, 20:404, :]
    im_mid = im_mid[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]

    if data_aug:
        if random.randint(0, 1):
            im_pre2 = np.fliplr(im_pre2)
            im_mid = np.fliplr(im_mid)
            im_pre1 = np.fliplr(im_pre1)
        if random.randint(0, 1):
            im_pre2 = np.flipud(im_pre2)
            im_mid = np.flipud(im_mid)
            im_pre1 = np.flipud(im_pre1)

    X0 = np.transpose(im_pre2,(2,0,1))
    X2 = np.transpose(im_mid, (2, 0, 1))

    y = np.transpose(im_pre1, (2, 0, 1))
    return X0.astype("float32")/ 255.0, \
            X2.astype("float32")/ 255.0,\
            y.astype("float32")/ 255.0

# Todo Add AIM dataloader
def AIM_Challenge_loader(root, im_clip_path, input_frame_size=(3,448, 448), output_frame_size = (3,256, 448), data_aug=True, middle=False, high_fps=False):
    """

    Args:
        root:
        im_clip_path:  im_clip_path contains 9 consecutive frames
        input_frame_size: the output frame shape, dataset shape is 3x720x1280
        data_aug: do transform

        if high_fps == true,
        we select

    Returns:

    """
    # first sample three consecutive frames [0, 2, 4, 6, 8, 10, 12, 14, 16]
    #                                        0  1  2  3  4   5   6   7  8
    len_clip = len(im_clip_path)
    randindex_list = []
    return_frameIndex = 1
    if data_aug:

        # Todo: do not use mixed 15fps, 30fps
        # mix 15 and 30 fps
        # if random.randint(0, 1):
        #     high_fps = False
        # else:
        #     high_fps = True
        if high_fps == False:  # input 15fps
            randindex = random.randint(0, len_clip-5)  # random select the five bunch {0,1,2,3,4}
            assert not randindex == 5  # Todo: check random range
            # random select the middle frames
            if middle == True:
                rand_mid_index = 2 + randindex  # mid frames {1,2,3}
            else:
                rand_mid_index = random.randint(1,3) + randindex   # mid frames {1,2,3}
            if random.randint(0, 1):  # reorder
                randindex_list =  [randindex, rand_mid_index, randindex+4]
                return_frameIndex = rand_mid_index - randindex - 1
            else:
                randindex_list =  [randindex+4, rand_mid_index, randindex]
                return_frameIndex = randindex - rand_mid_index + 3
        else: # input 30fps
            randindex = random.randint(0, len_clip-3)  # {0,1,2}
            #assert middle == False # not  middle model
            rand_mid_index = 1 + randindex # {1}
            if random.randint(0,1):
                randindex_list = [randindex, rand_mid_index, randindex + 2]
                return_frameIndex = 0
            else:
                randindex_list = [randindex + 2, rand_mid_index, randindex]
                return_frameIndex = 0



    if (middle == True) and high_fps == False:
        assert return_frameIndex == 1


    path_pre1 = im_clip_path[randindex_list[0]]
    path_mid = im_clip_path[randindex_list[1]]
    path_pre2 = im_clip_path[randindex_list[2]]


    im_pre2 = imread(path_pre2)  # HWVC
    im_pre1 = imread(path_pre1)
    im_mid = imread(path_mid)

    assert im_pre2.shape[0] <= 720
    assert im_pre2.shape[1] <= 1280

    # random crop
    h_offset = random.choice(range(720  - input_frame_size[1] +1))
    w_offset = random.choice(range(1280 - input_frame_size[2] +1))

    im_pre2 = im_pre2[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
    im_pre1 = im_pre1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
    im_mid  =  im_mid[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]

    # random flip
    if data_aug:
        if random.randint(0, 1):
            im_pre2 = np.fliplr(im_pre2)
            im_mid = np.fliplr(im_mid)
            im_pre1 = np.fliplr(im_pre1)

        # do not use up down
        # if random.randint(0, 1):
        #     im_pre2 = np.flipud(im_pre2)
        #     im_mid = np.flipud(im_mid)
        #     im_pre1 = np.flipud(im_pre1)

        # do not use rot
        # if input_frame_size[1] == input_frame_size[2]: # square
        #     if random.randint(0,1):
        #         im_pre2 = np.rot90(im_pre2,1)
        #         im_mid = np.rot90(im_mid,1)
        #         im_pre1 = np.rot90(im_pre1,1)
        #     if random.randint(0,1):
        #         im_pre2 = np.rot90(im_pre2,-1)
        #         im_mid = np.rot90(im_mid,-1)
        #         im_pre1 = np.rot90(im_pre1,-1)

    # X0 = np.transpose(im_pre2, (2, 0, 1))
    # X2 = np.transpose(im_mid, (2, 0, 1))
    #
    # y = np.transpose(im_pre1, (2, 0, 1))
    X0 = np.transpose(im_pre1, (2, 0, 1))
    X2 = np.transpose(im_pre2, (2, 0, 1))
    y = np.transpose(im_mid, (2, 0, 1))

    return X0.astype("float32") / 255.0, \
           X2.astype("float32") / 255.0, \
           y.astype("float32") / 255.0, \
           return_frameIndex


class ListDataset_random_mid(data.Dataset):
    def __init__(self, root, path_list,  loader=default_loader_random_mid, middle=False,high_fps=False): #transform=None, target_transform=None, co_transform=None,

        self.root = root
        self.path_list = path_list
        # self.transform = transform
        # self.target_transform = target_transform
        # self.co_transform = co_transform
        self.loader = loader
        self.middle = middle
        self.high_fps = high_fps

    def __getitem__(self, index):
        path = self.path_list[index]
        # print(path)
        image_0,image_2,image_1, frame_index = self.loader(self.root, path, middle=self.middle, high_fps=self.high_fps)
        # if self.co_transform is not None:
        #     inputs, target = self.co_transform(image_0,image_2,image_1)
        # if self.transform is not None:
        #     image_0 = self.transform(image_0)
        #     image_2 = self.transform(image_2)
        # if self.target_transform is not None:
        #     image_1 = self.target_transform(image_1)
        return image_0,image_2,image_1, frame_index

    def __len__(self):
        return len(self.path_list)

class ListDataset(data.Dataset):
    def __init__(self, root, path_list,  loader=default_loader): #transform=None, target_transform=None, co_transform=None,

        self.root = root
        self.path_list = path_list
        # self.transform = transform
        # self.target_transform = target_transform
        # self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.path_list[index]
        # print(path)
        image_0,image_2,image_1 = self.loader(self.root, path)
        # if self.co_transform is not None:
        #     inputs, target = self.co_transform(image_0,image_2,image_1)
        # if self.transform is not None:
        #     image_0 = self.transform(image_0)
        #     image_2 = self.transform(image_2)
        # if self.target_transform is not None:
        #     image_1 = self.target_transform(image_1)
        return image_0,image_2,image_1

    def __len__(self):
        return len(self.path_list)
