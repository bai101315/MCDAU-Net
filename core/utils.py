from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
from models import *
import pickle as pkl
from torch.autograd import Variable
import imageio
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import json

# 数据集路径
BIGPCHASEDB1DataTrainPath = './data/96_CHASEDB1/PCHASEDB1_train'
BIGPCHASEDB1DataTestPath = './data/96_CHASEDB1/PCHASEDB1_test'
BIGPDRIVEDataTrainPath = './data/96_DRIVE/PDRIVE_train'
BIGPDRIVEDataTestPath = './data/96_DRIVE/PDRIVE_test'
BIGPSTAREDataTrainPath = './data/96_STARE/PSTARE_train'
BIGPSTAREDataTestPath = './data/96_STARE/PSTARE_test'

def get_data(dataset, img_name, img_size=256, gpu=True, flag='train'):

    def get_label(label):
        tmp_gt = label.copy()
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()


        return label, tmp_gt

    images = []
    imageGreys = []
    labels = []
    tmp_gts = []

    img_shape = []
    label_ori = []
    Mask_ori = []
    batch_size = len(img_name)

    for i in range(batch_size):

        if dataset == "BIG_DRIVE":
            if flag == 'train':
                img_path = os.path.join(BIGPDRIVEDataTrainPath, 'images', img_name[i].rstrip('\n'))

                label_name = img_name[i].rstrip('\n')[:5] + 'training.png'
                mask_name = img_name[i].rstrip('\n')[:5] + 'training.png'

                mask_path = os.path.join(BIGPDRIVEDataTrainPath, 'masks', mask_name)
                # FOVmask = imageio.mimread(mask_path)
                FOVmask = cv2.imread(mask_path)

                if FOVmask is not None:
                    FOVmask = np.array(FOVmask)
                    FOVmask = FOVmask[0]

                label_path = os.path.join(BIGPDRIVEDataTrainPath, 'labels', label_name)
                img = cv2.imread(img_path)
                # label = imageio.mimread(label_path)
                label = cv2.imread(label_path)
                if label is not None:
                    # 读取的方式不对
                    label = np.array(label)
                    label = label[:, :, 0]
                    label = label.reshape(96, 96)
                    # label = label[3]
            else:
                img_path = os.path.join(BIGPDRIVEDataTestPath, 'images', img_name[i].rstrip('\n'))
                # label_name = img_name[i].rstrip('\n')[:8] + 'test.png'
                # mask_name = img_name[i].rstrip('\n')[:8] + 'test.png'
                label_name = img_name
                mask_name = img_name

                mask_path = os.path.join(BIGPDRIVEDataTestPath, 'masks', mask_name[i].rstrip('\n'))
                # FOVmask = imageio.mimread(mask_path)
                FOVmask = cv2.imread(mask_path)

                if FOVmask is not None:
                    FOVmask = np.array(FOVmask)
                    FOVmask = FOVmask[0]

                label_path = os.path.join(BIGPDRIVEDataTestPath, 'labels', label_name[i].rstrip('\n'))
                img = cv2.imread(img_path)

                # label = imageio.mimread(label_path)
                label = cv2.imread(label_path)
                if label is not None:
                    # 读取的方式不对
                    label = np.array(label)
                    label = label[:, :, 0]
                    label = label.reshape(96, 96)
                    # label = label[3]

        if dataset == "BIG_STARE":
            if flag == 'train':
                img_path = os.path.join(BIGPSTAREDataTrainPath, 'images', img_name[i].rstrip('\n'))

                label_name = img_name[i].rstrip('\n')[:5] + 'training.png'
                mask_name = img_name[i].rstrip('\n')[:5] + 'training.png'

                mask_path = os.path.join(BIGPSTAREDataTrainPath, 'masks', mask_name)
                # FOVmask = imageio.mimread(mask_path)
                FOVmask = cv2.imread(mask_path)

                if FOVmask is not None:
                    FOVmask = np.array(FOVmask)
                    FOVmask = FOVmask[0]

                label_path = os.path.join(BIGPSTAREDataTrainPath, 'labels', label_name)
                img = cv2.imread(img_path)
                # label = imageio.mimread(label_path)
                label = cv2.imread(label_path)
                if label is not None:
                    # 读取的方式不对
                    label = np.array(label)
                    label = label[:, :, 0]
                    label = label.reshape(img_size, img_size)
                    # label = label[3]

            else:
                img_path = os.path.join(BIGPSTAREDataTestPath, 'images', img_name[i].rstrip('\n'))
                # label_name = img_name[i].rstrip('\n')[:8] + 'test.png'
                # mask_name = img_name[i].rstrip('\n')[:8] + 'test.png'
                label_name = img_name
                mask_name = img_name

                mask_path = os.path.join(BIGPSTAREDataTestPath, 'masks', mask_name[i].rstrip('\n'))
                # FOVmask = imageio.mimread(mask_path)
                FOVmask = cv2.imread(mask_path)

                if FOVmask is not None:
                    FOVmask = np.array(FOVmask)
                    FOVmask = FOVmask[0]

                label_path = os.path.join(BIGPSTAREDataTestPath, 'labels', label_name[i].rstrip('\n'))
                img = cv2.imread(img_path)

                # label = imageio.mimread(label_path)
                label = cv2.imread(label_path)
                if label is not None:
                    # 读取的方式不对
                    label = np.array(label)
                    label = label[:, :, 0]
                    label = label.reshape(img_size, img_size)
                    # label = label[3]


        if dataset == "BIG_CHASEDB1":
            if flag == 'train':
                img_path = os.path.join(BIGPCHASEDB1DataTrainPath, 'images', img_name[i].rstrip('\n'))

                label_name = img_name[i].rstrip('\n')[:5] + 'training.png'
                mask_name = img_name[i].rstrip('\n')[:5] + 'training.png'

                mask_path = os.path.join(BIGPCHASEDB1DataTrainPath, 'masks', mask_name)
                # FOVmask = imageio.mimread(mask_path)
                FOVmask = cv2.imread(mask_path)

                if FOVmask is not None:
                    FOVmask = np.array(FOVmask)
                    FOVmask = FOVmask[0]

                label_path = os.path.join(BIGPCHASEDB1DataTrainPath, 'labels', label_name)
                img = cv2.imread(img_path)
                # label = imageio.mimread(label_path)
                label = cv2.imread(label_path)
                if label is not None:
                    # 读取的方式不对
                    label = np.array(label)
                    label = label[:, :, 0]
                    label = label.reshape(img_size, img_size)
                    # label = label[3]
            else:
                img_path = os.path.join(BIGPCHASEDB1DataTestPath, 'images', img_name[i].rstrip('\n'))
                # label_name = img_name[i].rstrip('\n')[:8] + 'test.png'
                # mask_name = img_name[i].rstrip('\n')[:8] + 'test.png'
                label_name = img_name
                mask_name = img_name

                mask_path = os.path.join(BIGPCHASEDB1DataTestPath, 'masks', mask_name[i].rstrip('\n'))
                # FOVmask = imageio.mimread(mask_path)
                FOVmask = cv2.imread(mask_path)

                if FOVmask is not None:
                    FOVmask = np.array(FOVmask)
                    FOVmask = FOVmask[0]

                label_path = os.path.join(BIGPCHASEDB1DataTestPath, 'labels', label_name[i].rstrip('\n'))
                img = cv2.imread(img_path)

                # label = imageio.mimread(label_path)
                label = cv2.imread(label_path)
                if label is not None:
                    # 读取的方式不对
                    label = np.array(label)
                    label = label[:, :, 0]
                    label = label.reshape(img_size, img_size)
                    # label = label[3]


        img_shape.append(img.shape)
        label_ori.append(label)

        label[label == 255] = 1

        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(FOVmask, (img_size, img_size), interpolation=cv2.INTER_AREA)



        segmap = SegmentationMapsOnImage(label, shape=label.shape)
        seq = iaa.Sequential([
            # iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
            # iaa.Sharpen((0.0, 1.0)),  # sharpen the image
            iaa.Affine(rotate=(-20, 20)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            # iaa.Flipud(0.2),
            # iaa.ElasticTransformation(alpha =50, sigma=5)  # apply water effect (affects segmaps)
            iaa.GammaContrast((0.5, 2.0))
            # iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.5))),
        ], random_order=True)

        # iage_aug = rotate.augment_image(img)
        if flag == 'train':
            img, label = seq(image=img, segmentation_maps=segmap)
            label = np.squeeze(label.arr)


        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgGrey = imgGrey[np.newaxis, :, :]

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()
        imgGrey = Variable(torch.from_numpy(imgGrey)).double()

        if gpu:
            img = img.cuda()
            imgGrey = imgGrey.cuda()

        label, tmp_gt = get_label(label)

        images.append(img)
        labels.append(label)

        tmp_gts.append(tmp_gt)
        imageGreys.append(imgGrey)
        Mask_ori.append(mask)

    images = torch.stack(images)
    imageGreys = torch.stack(imageGreys)

    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)

    if flag:
        label_ori = np.stack(label_ori)

    return images, imageGreys, labels, tmp_gts, img_shape, label_ori, Mask_ori


def calculate_Accuracy(confusion):
    EPS = 1e-12
    confusion = np.asarray(confusion)

    pos = np.sum(confusion, 1).astype(np.float32)  # 1 for row
    res = np.sum(confusion, 0).astype(np.float32)  # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)
    meanIU = np.mean(IU)
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1] + confusion[0][1] + EPS)
    Sp = confusion[0][0] / (confusion[0][0] + confusion[1][0] + EPS)
    return meanIU, Acc, Se, Sp, IU


def get_model(model_name):
    if model_name == 'MCDAU_Net':
        return MCDAU_Net

def read_spilt_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    image_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(image_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for cla in image_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    # test_images_path是所有图片的路径
    return test_images_path

# dataset_list = ['DRIVE', "STARE", "CHASEDB1"]
def get_img_list(dataset, SubID, flag='train'):

    if dataset == "BIG_DRIVE":
        if flag == 'train':
            with open(os.path.join(BIGPDRIVEDataTrainPath, "PDRIVEtraining.txt"), 'r') as f:
                img_list = f.readlines()
        else:
            root = './data/96_DRIVE/PDRIVE_test/images'
            img_list = read_spilt_data(root)

    if dataset == "BIG_STARE":
        if flag == 'train':
            with open(os.path.join(BIGPSTAREDataTrainPath, "PSTAREtraining.txt"), 'r') as f:
                img_list = f.readlines()
        else:
            root = './data/96_STARE/PSTARE_test/images'
            img_list = read_spilt_data(root)

    if dataset == "BIG_CHASEDB1":
        if flag == 'train':
            with open(os.path.join(BIGPCHASEDB1DataTrainPath, "PCHASEDB1training.txt"), 'r') as f:
                img_list = f.readlines()
        else:
            root = './data/96_CHASEDB1/PCHASEDB1_test/images'
            img_list = read_spilt_data(root)

    return img_list