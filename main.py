import torch
import torch.nn as nn

import sklearn.metrics as metrics

import os
import argparse
import time
from core.utils import calculate_Accuracy, get_img_list, get_model, get_data
from pylab import *
import random
from test import fast_test

plt.switch_backend('agg')
import cv2
from thop import profile
from thop import clever_format
import torch.nn.functional as F

# --------------------------------------------------------------------------------

models_list = ["MCDAU_Net"]
dataset_list = ["BIG_DRIVE", "BIG_CHASEDB1", "BIG_STARE"]

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch')

parser.add_argument('--epochs', type=int, default=100,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class')
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
parser.add_argument('--SubImageID', type=int, default=20,
                    help='Only for Stare Dataset')
parser.add_argument('--model_id', type=int, default=0,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=20,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=96,
                    help='the train img size')
parser.add_argument('--my_description', type=str, default='',
                    help='some description define your train')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=False,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='0',
                    help='the gpu used')

args = parser.parse_args()
print(args)
# --------------------------------------------------------------------------------

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

RootDir = os.getcwd()
model_name = models_list[args.model_id]
Dataset = dataset_list[args.datasetID]
SubID = args.SubImageID
model = get_model(model_name)

model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if args.use_gpu:
    model.cuda()
    print('GPUs used: (%s)' % args.gpu_avaiable)
    print('------- success use GPU --------')

EPS = 1e-12
# define path
img_list = get_img_list(Dataset, SubID, flag='train')
test_img_list = get_img_list(Dataset, SubID, flag='test')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
criterion = nn.NLLLoss2d()
softmax_2d = nn.Softmax2d()

IOU_best = 0

print('This model is %s_%s_%s_%s' % (model_name, args.n_class, args.img_size, args.my_description))
if not os.path.exists(r'%s-models-%s_%s' % (RootDir, model_name, args.my_description)):
    os.mkdir(r'%s-models-%s_%s' % (RootDir, model_name, args.my_description))

with open(r'%s-logs-%s_%s.txt' % (RootDir, model_name, args.my_description), 'w+') as f:
    f.write('This model is %s_%s: ' % (model_name, args.my_description) + '\n')
    f.write('args: ' + str(args) + '\n')
    f.write('train lens: ' + str(len(img_list)) + ' | test lens: ' + str(len(test_img_list)))
    f.write('\n\n---------------------------------------------\n\n')

BestAccuracy = 0
BestSe = 0
for epoch in range(args.epochs):
    model.train()
    begin_time = time.time()
    print('This model is %s_%s_%s_%s' % (
        model_name, args.n_class, args.img_size, args.my_description))
    random.shuffle(img_list)

    if (epoch % 50 == 0) and epoch != 0:
        # args.lr /= 10

        power = 0.9
        lr=args.lr*(1-epoch/args.epochs)**power

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i, (start, end) in enumerate(zip(range(0, len(img_list), args.batch_size),
                                         range(args.batch_size, len(img_list) + args.batch_size,
                                               args.batch_size))):

        path = img_list[start:end]
        img_96, imageGreys_96, gt_96, tmp_gt_96, img_shape_96, label_ori_96, mask_ori_96 = get_data(Dataset, path, img_size=args.img_size, gpu=args.use_gpu)

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

        optimizer.zero_grad()
        out_96 = model(img_96)
        out_96 = torch.log(softmax_2d(out_96) + EPS)
        loss1 = criterion(out_96, gt_96)

        out = model(img)
        out = torch.log(softmax_2d(out) + EPS)
        loss2 = criterion(out, gt)

        (loss1 + loss2).backward()
        optimizer.step()
        ppi = np.argmax(out.cpu().data.numpy(), 1)
        ppi_96 = np.argmax(out_96.cpu().data.numpy(), 1)

        ppi = np.ceil((ppi + ppi_96[:, 16:80, 16:80]) / 2)
        # ppi = np.ceil((ppi + ppi_96[:, 16:48, 16:48]) / 2)

        # Select the pixel inside FOV
        tmp_out = ppi.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])
        Mask = mask_ori_96[0]
        Mask = Mask[16:80, 16:80]
        # Mask = Mask[16:48, 16:48]

        # cv2.imshow('img', Mask*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        Mask = Mask.reshape([-1])
        SelectOut = tmp_out[np.flatnonzero(Mask)]
        SelectGT = tmp_gt[np.flatnonzero(Mask)]

        my_confusion = metrics.confusion_matrix(SelectOut, SelectGT).astype(np.float32)

        [m, n] = my_confusion.shape

        if m <= 1 or n <= 1:
            continue
        meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)

        print('training finish, time: %.1f s' % (time.time() - begin_time))

        print(str('model: {:s}_{:s} | epoch_batch: {:d}_{:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}'
                  '| Background_IOU: {:f}, vessel_IOU: {:f}').format(model_name, args.my_description, epoch, i,
                                                                     (loss1 + loss2).item(), Acc, Se, Sp,
                                                                     IU[0], IU[1]))
    if epoch %5 == 0 and epoch !=0 :
        Se_tmp = fast_test(model, args, test_img_list, model_name)
        print('BestSe:', BestSe)
        if Se_tmp > BestSe:
            BestSe = Se_tmp

            save_path = './pth'
            save_path = os.path.join(save_path, '%s_%s_%s.pth' % (model_name, args.my_description, str(epoch)))

            print(save_path)
            torch.save(model.state_dict(), save_path)
            # For evaluation
            print('success save Nucleus_best model')
