# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

import os
import argparse
import time
from core.utils import calculate_Accuracy, get_model, get_data, get_img_list
from pylab import *
import cv2
import warnings
warnings.filterwarnings("ignore")
plt.switch_backend('agg')

# --------------------------------------------------------------------------------
models_list = ["MCDAU_Net"]
dataset_list = ["BIG_DRIVE", "BIG_CHASEDB1", "BIG_STARE"]
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')
# ---------------------------
# params do not need to change
# ---------------------------
parser.add_argument('--epochs', type=int, default=250,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
# ---------------------------
# model
# ---------------------------
parser.add_argument('--datasetID', type=int, default=0,
                    help='dir of the all img')
parser.add_argument('--SubImageID', type=int, default=1,
                    help='Only for Stare Dataset')

parser.add_argument('--best_model', type=str,  default='C:\\Users\\BAI\\Desktop\\DeepGuidance-main\\32.pth',
                    help='the pretrain model')

parser.add_argument('--model_id', type=int, default=0,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=40,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=96,
                    help='the train img size')
parser.add_argument('--my_description', type=str, default='test data aug',
                    help='some description define your train')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='0',
                    help='the gpu used')

args = parser.parse_args()

def fast_test(model, args, img_list, model_name):
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12

    Dataset = dataset_list[args.datasetID]
    SubID = args.SubImageID

    Background_IOU = []
    Vessel_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []

    for i, path in enumerate(img_list):
        img_96, imageGreys_96, gt_96, tmp_gt_96, img_shape_96, label_ori_96, mask_ori_96 = get_data(Dataset, [path], img_size=args.img_size, gpu=args.use_gpu,flag='test')

        img = img_96[:, :, 16:80, 16:80]
        imageGreys = imageGreys_96[:, :, 16:80, 16:80]
        gt = gt_96[:, 16:80, 16:80]
        tmp_gt = tmp_gt_96[:, 16:80, 16:80]
        label_ori = label_ori_96[:, 16:80, 16:80]

        # img = img_96[:, :, 16:48, 16:48]
        # imageGreys = imageGreys_96[:, :, 16:48, 16:48]
        # gt = gt_96[:, 16:48, 16:48]
        # tmp_gt = tmp_gt_96[:, 16:48, 16:48]
        # label_ori = label_ori_96[:, 16:48, 16:48]


        # Save the image
        model.eval()
        start = time.time()
        # 保存out
        out = model(img)
        out_96 = model(img_96)
        end = time.time()

        out = torch.log(softmax_2d(out) + EPS)
        out_96 = torch.log(softmax_2d(out_96) + EPS)

        out = F.upsample(out, size=(64, 64), mode='bilinear')
        # out = F.upsample(out, size=(32, 32), mode='bilinear')
        out = out.cpu().data.numpy()
        y_pred = out[:, 1, :, :]
        y_pred = y_pred.reshape([-1])
        ppi = np.argmax(out, 1)

        out_96 = F.upsample(out_96, size=(96, 96), mode='bilinear')
        # out_96 = F.upsample(out_96, size=(64, 64), mode='bilinear')
        out_96 = out_96.cpu().data.numpy()
        y_pred_96 = out_96[:, 1, :, :]
        y_pred_96 = y_pred_96.reshape([-1])
        ppi_96 = np.argmax(out_96, 1)
        ppi = np.ceil((ppi + ppi_96[:, 16:80, 16:80]) / 2)
        # ppi = np.ceil((ppi + ppi_96[:, 16:48, 16:48]) / 2)

        # Output the prediction
        # id = path.split('\\')[0]
        # image_name = path[6:]
        # root = os.getcwd()
        # save_dir = os.path.join(root, 'result', id)
        # if not os.path.exists(r'%s' % (save_dir)):
        #     os.makedirs(r'%s' % (save_dir))
        #
        # temp_dir = save_dir
        # save_dir = os.path.join(temp_dir, image_name)
        # # print(save_dir)
        # # 保存label
        # # ImageName = path[:-4] + 'png'
        # # gtName = os.path.join(gt_root_dir, ImageName)
        # ppi_temp = np.squeeze(ppi)
        # cv2.imwrite(save_dir, ppi_temp * 255)

        tmp_out = ppi.reshape([-1])
        tmp_gt = label_ori.reshape([-1])
        Mask = mask_ori_96[0]
        Mask = Mask[16:80, 16:80]
        # Mask = Mask[16:48, 16:48]

        Mask = Mask.reshape([-1])
        if np.all(Mask == 0):
            continue

        SelectOut = tmp_out[np.flatnonzero(Mask)]
        SelectGT = tmp_gt[np.flatnonzero(Mask)]

        my_confusion = metrics.confusion_matrix(SelectOut, SelectGT).astype(np.float32)

        if np.all(tmp_gt == 0) or np.all(y_pred == 0):
            continue

        [m, n] = my_confusion.shape
        if m <= 1 or n <= 1:
            continue

        meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)
        Auc = roc_auc_score(tmp_gt, y_pred)
        AUC.append(Auc)

        Background_IOU.append(IU[0])
        Vessel_IOU.append(IU[1])
        ACC.append(Acc)
        SE.append(Se)
        SP.append(Sp)

    print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s ' % (str(np.mean(np.stack(ACC))), str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))

    # store test information

    RootDir = os.getcwd()
    with open(r'%s-logs-%s_%s.txt' % (RootDir, model_name, args.my_description), 'a+') as f:
        f.write('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))
        f.write('\n\n')

    #return np.mean(np.stack(Vessel_IOU))
    return np.mean(np.stack(SE))

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

    model_name = models_list[args.model_id]

    model = get_model(model_name)
    model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)

    if args.use_gpu:
        model.cuda()
    if True:
        model_path = "./pth/UNet_center_Dense_CCAF__5.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=False)
        print('success load models: %s_%s' % (model_name, args.my_description))

    print ('This model is %s_%s_%s' % (model_name, args.n_class, args.img_size))
    Dataset = dataset_list[args.datasetID]
    SubID = args.SubImageID

    test_img_list = get_img_list(Dataset, SubID, flag='test')

    fast_test(model, args, test_img_list, model_name)
