import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from core.utils import calculate_Accuracy, get_img_list
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import imageio

import cv2

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
    F1 = 2 * confusion[1][1] / (2 * confusion[1][1] + confusion[0][1] + confusion[1][0] + EPS)

    return meanIU, Acc, Se, Sp, IU, F1

def test():

    IMAGE_SAVE_PATH = './test/result'
    # DRIVE
    # data_root = 'C:\\Users\BAI\Desktop\psp\DRIVE\\test'
    # softmax_2d = nn.Softmax2d()
    # img_names = [i for i in os.listdir(os.path.join(data_root, "result")) if i.endswith("test.tif")]
    # img_list = [os.path.join(data_root, "result", i) for i in img_names]
    # mask_names = [i for i in os.listdir(os.path.join(data_root, "masks")) if i.endswith("_test_mask.gif")]
    # mask_list = [os.path.join(data_root, "masks", i) for i in mask_names]
    # label_names = [i for i in os.listdir(os.path.join(data_root, "label")) if i.endswith(".tif")]
    # label_list = [os.path.join(data_root, "label", i) for i in label_names]

    # CHASEDB1
    data_root = IMAGE_SAVE_PATH[:6]
    print(data_root)
    softmax_2d = nn.Softmax2d()
    img_names = [i for i in os.listdir(os.path.join(data_root, "result")) if i.endswith("test.tif")]
    img_list = [os.path.join(data_root, "result", i) for i in img_names]
    mask_names = [i for i in os.listdir(os.path.join(data_root, "masks")) if i.endswith("tif")]
    mask_list = [os.path.join(data_root, "masks", i) for i in mask_names]
    label_names = [i for i in os.listdir(os.path.join(data_root, "label")) if i.endswith(".tif")]
    label_list = [os.path.join(data_root, "label", i) for i in label_names]

    EPS = 1e-12

    Background_IOU = []
    Vessel_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []
    F1 = []

    # CHASEDB1
    # for i in range(0,8):
    # Stare
    # for i in range(0,20):
    for i in range(0, 5):
        img_name = img_list[i]
        label_name = label_list[i]
        mask_name = mask_list[i]
        # print(img_list[i])

        ppi = cv2.imread(img_name)
        ppi = ppi[:, :, 0]

        y_pred = ppi / 255
        y_pred = y_pred.reshape([-1])

        label_ori = imageio.mimread(label_name)
        label_ori = np.array(label_ori)
        label_ori = label_ori[0]
        label_ori = label_ori[:, :, 0]

        mask_ori = imageio.mimread(mask_name)
        mask_ori = np.array(mask_ori)
        mask_ori = mask_ori[0]
        mask_ori = mask_ori[:, :, 0]

        # cv2.namedWindow('image')
        # cv2.imshow('image',ppi)
        # cv2.waitKey(0)

        tmp_out = ppi.reshape([-1])
        tmp_gt = label_ori.reshape([-1])
        Mask = mask_ori
        Mask = Mask.reshape([-1])

        # np.set_printoptions(threshold=np.inf)
        # print(Mask)

        SelectOut = tmp_out[np.flatnonzero(Mask)]
        SelectGT = tmp_gt[np.flatnonzero(Mask)]

        my_confusion = metrics.confusion_matrix(SelectOut, SelectGT).astype(np.float32)
        meanIU, Acc, Se, Sp, IU, f1 = calculate_Accuracy(my_confusion)

        Auc = roc_auc_score(tmp_gt, y_pred)
        AUC.append(Auc)

        Background_IOU.append(IU[0])
        Vessel_IOU.append(IU[1])
        ACC.append(Acc)
        SE.append(Se)
        SP.append(Sp)
        F1.append(f1)

        print(
            str('F1: {:.3f}|Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}| | Auc: {:.3f}| Background_IOU: {:f}, vessel_IOU: {:f}').format(
                f1, Acc, Se, Sp, Auc, IU[0], IU[1]))

    print("最终结果：")
    print('F1: %s |Acc: %s | Se: %s | Sp: %s | Auc: %s | Background_IOU: %s | vessel_IOU: %s ' %
          (str(np.mean(np.stack(F1))), str(np.mean(np.stack(ACC))), str(np.mean(np.stack(SE))),
           str(np.mean(np.stack(SP))), str(np.mean(np.stack(AUC))), str(np.mean(np.stack(Background_IOU))),
           str(np.mean(np.stack(Vessel_IOU)))))

