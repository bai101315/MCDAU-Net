import functools
import os

import numpy as np
import pandas as pd
import torch.nn as nn
from skimage import io, transform, exposure, filters, color

from core.utils import get_img_list


def read_df(fpath, data_dir=''):
    # Dataset = 'DRIVE'
    # Dataset = 'CHASEDB1'
    Dataset = 'STARE'

    img_list = get_img_list(Dataset, 'Fold2', flag='train')
    test_img_list = get_img_list(Dataset, 'Fold2', flag='test')
    #
    # img_list=get_img_list(Dataset,12,flag='train')
    # test_img_list=get_img_list(Dataset,12,flag='test')

    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'data', Dataset)

    x_paths = []
    y_paths = []
    z_paths = []

    for image_name in test_img_list:
        # for image_name in img_list:
        if Dataset == "DRIVE":
            image_id = image_name.split('_')[0]

            # x_path = os.path.join(base_path,'images',image_id+'_test.tif')
            # y_path = os.path.join(base_path,'label',image_id + '_manual1.gif')
            # z_path = os.path.join(base_path,'masks',image_id + '_test_mask.gif')

            x_path = os.path.join(base_path, 'images', image_id + '_training.tif')
            y_path = os.path.join(base_path, 'label', image_id + '_manual1.gif')
            z_path = os.path.join(base_path, 'masks', image_id + '_training_mask.gif')

        elif Dataset == "CHASEDB1":
            # print(image_name)

            image_id = image_name.rstrip('\n')[:-4]
            x_path = os.path.join(base_path, 'images', image_id + '.jpg')
            y_path = os.path.join(base_path, 'label', image_id + '_1stHO.png')
            z_path = os.path.join(base_path, 'Masks', image_id + '.jpg')

        elif Dataset == 'STARE':
            image_id = image_name.rstrip('\n')[0:6]

            x_path = os.path.join(base_path, 'images', image_id + '.ppm')
            y_path = os.path.join(base_path, 'labels', image_id + '.ah.ppm')
            z_path = os.path.join(base_path, 'Masks', image_id + '.jpg')

        x_paths.append(x_path)
        y_paths.append(y_path)
        z_paths.append(z_path)

    x_paths = pd.Series(x_paths)
    y_paths = pd.Series(y_paths)
    z_paths = pd.Series(z_paths)

    return x_paths, y_paths, z_paths


def _process_pathnames(fname, lname, mname, resize=None):
    img = io.imread(fname)
    gt = io.imread(lname)
    mask = io.imread(mname)

    if gt.ndim < 3:
        gt = np.expand_dims(gt, -1)
    gt = gt[..., :1]
    gt = (gt > 0).astype(int)  # binarize the ground-truth

    if mask.ndim < 3:
        mask = np.expand_dims(mask, -1)
    mask = mask[..., :1]

    mask[mask > 240] = 255
    mask[mask < 240] = 0
    mask = (mask > 0).astype(int)  # binarize the ground-truth

    if resize is not None:
        img = transform.resize(img, resize)

        gt = transform.resize(gt, resize)
        gt = gt >= filters.threshold_otsu(gt)

        mask = transform.resize(mask, resize)
        mask = mask >= filters.threshold_otsu(mask)

    return img, gt, mask


### Data augmentation routines
def shift_img(img, gt, mask, width_shift_range, height_shift_range, rotate_range):
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = np.random.uniform(-width_shift_range * img.shape[1],
                                                  width_shift_range * img.shape[1])
        if height_shift_range:
            height_shift_range = np.random.uniform(-height_shift_range * img.shape[0],
                                                   height_shift_range * img.shape[0])
        tr = transform.AffineTransform(translation=(width_shift_range, height_shift_range))
        img = transform.warp(img, tr, preserve_range=True)
        gt = transform.warp(gt, tr, preserve_range=True)
        mask = transform.warp(mask, tr, preserve_range=True)

    if rotate_range:
        if isinstance(rotate_range, np.ScalarType):
            degre = np.random.uniform(-rotate_range, rotate_range)
        else:
            degre = np.random.uniform(rotate_range[0], rotate_range[1])
        img = transform.rotate(img, degre, preserve_range=True)
        gt = transform.rotate(gt, degre, preserve_range=True)
        mask = transform.rotate(mask, degre, preserve_range=True)

    return img, gt, mask


def flip_img(img, gt, mask, horizontal_flip, vertical_flip):
    if horizontal_flip:
        flip_prob = np.random.uniform(0.0, 1.0)
        img, gt, mask = (img, gt, mask) if flip_prob >= 0.5 else (np.flip(img, 1), np.flip(gt, 1), np.flip(mask, 1))
    if vertical_flip:
        flip_prob = np.random.uniform(0.0, 1.0)
        img, gt, mask = (img, gt, mask) if flip_prob >= 0.5 else (np.flip(img, 0), np.flip(gt, 0), np.flip(mask, 0))
    return img, gt, mask


def _process_imgt(img, gt, mask, gamma=0, gray=False, xyz=False, hed=False, green=True,
                  horizontal_flip=False, width_shift_range=0, clahe=False,
                  height_shift_range=0, vertical_flip=False, rotate_range=(0, 0), bw_gt=True, bw_mask=True):
    # img = cv2.imread("C:\\Users\BAI\Desktop\data\DRIVE\images\\20_test.tif")
    img = exposure.rescale_intensity(img.astype(float), out_range=(0, 1))
    if green:
        img = img[:, :, 1]
        # result_patch_path = './result/green.png'
        # io.imsave(result_patch_path, 255*np.squeeze(img))
        # plot_images([np.squeeze(img), np.squeeze(img)], title="img")  # 绘制图像
        # plot_images(np.squeeze(img), title="img")  # 绘制图像
    if gray:
        img = color.rgb2gray(img)
    if xyz:
        img = color.rgb2xyz(img)
    if hed:
        img = color.rgb2hed(img)
    img = exposure.rescale_intensity(img, out_range=(0, 1))
    if clahe:
        img = exposure.equalize_adapthist(img)

    if gamma:
        img = exposure.adjust_gamma(img, gamma)
        img = exposure.rescale_intensity(img, out_range=(0, 1))

        # result_patch_path = './result/test.png'
        # io.imsave(result_patch_path, 255*np.squeeze(img))
        # plot_images([np.squeeze(img), np.squeeze(img)], title="img")  # 绘制图像

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    img, gt, mask = flip_img(img, gt, mask, horizontal_flip, vertical_flip)
    img, gt, mask = shift_img(img, gt, mask, width_shift_range, height_shift_range, rotate_range)

    if bw_gt:
        gt = gt >= filters.threshold_otsu(gt)
    if bw_mask:
        mask = mask >= filters.threshold_otsu(mask)

    return img, gt, mask


def fixed_patch_ids_creation(im_paths, gt_paths, mask_paths, spatial_shape=None,
                             p_stride=16, shuffle=True, per_label=0, mask=None):
    all_ids = []
    mask = mask if mask is not None else 1
    for im_path, gt_path, mask_path in zip(im_paths, gt_paths, mask_paths):
        if p_stride > 0:
            ids = np.zeros(spatial_shape, dtype='int')
            if ids.ndim == 2:
                ids[0::p_stride, 0::p_stride] = 1
            else:
                ids[0::p_stride, 0::p_stride, 0::p_stride] = 1
            ids = ids * mask
            ids = np.array(np.nonzero(ids)).T
            n = len(ids)
            ap = np.c_[
                np.expand_dims([im_path] * n, -1), np.expand_dims([gt_path] * n, -1), np.expand_dims([mask_path] * n,
                                                                                                     -1), ids]
            all_ids.extend(ap)

        if per_label > 0:
            # Adding samples based on the classes distribution.
            _, gt, mask = _process_pathnames(im_path, gt_path, mask_path, resize=spatial_shape)
            cls_ids = []
            for c in np.unique(gt):
                search_area = np.nonzero((np.squeeze(gt) == c) * mask)
                if len(search_area[0]) == 0:
                    continue
                search_area = np.array(search_area).T
                search_area = np.random.permutation(search_area)
                cls_ids.append(search_area[:per_label])
            cls_ids = np.concatenate([x for x in cls_ids])
            n = len(cls_ids)
            ap = np.c_[[im_path] * n, [gt_path] * n, [mask_path] * n, cls_ids]
            all_ids.extend(ap)

    all_ids = np.array(all_ids)
    if shuffle:
        np.random.shuffle(all_ids)

    return all_ids


# class Patch_Sequence(tf.keras.utils.Sequence):
class Patch_Sequence(nn.Module):
    def __init__(self, fixed_patch_ids, p_shape=(32, 32, 3),
                 reader_fn=functools.partial(_process_pathnames),
                 preproc_fn=functools.partial(_process_imgt),
                 batch_size=32,
                 MAX_IM_QUEUE=20, unsup=False, resize=None):
        self.ids = fixed_patch_ids  #
        self.p_shape = p_shape
        self.batch_size = batch_size
        self.reader_fn = reader_fn
        self.preproc_fn = preproc_fn
        self.MAX_IM_QUEUE = MAX_IM_QUEUE
        self.im_stack = {}
        self.unsup = unsup
        self.resize = resize

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def __getitem__(self, idx):

        # 这张图片的image，label，mask
        cur_id = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []
        batch_z = []
        batch_place = []

        for pos in cur_id:

            pid, pim, pgt, pma = pos[3:], pos[0], pos[1], pos[2]
            x_p, y_p = pid.astype(int)
            hash_im = hash(pim)
            # 自己注释啊掉的
            # if not self.im_stack.has_key(hash_im):

            # print(x_p)
            # print(y_p)
            # print(self.ids)

            if hash_im not in self.im_stack.keys():
                img, gt, mask = self.reader_fn(pim, pgt, pma)
                img, gt, mask = self.preproc_fn(img, gt, mask)

                # 96数据集必须的
                img = np.pad(img, ((16, 16), (16, 16), (0, 0)), 'constant')
                gt = np.pad(gt, ((16, 16), (16, 16), (0, 0)), 'constant')
                mask = np.pad(mask, ((16, 16), (16, 16), (0, 0)), 'constant')

                if len(self.im_stack.keys()) > self.MAX_IM_QUEUE:
                    self.im_stack.popitem()
                self.im_stack[hash_im] = (img, gt, mask)
            else:
                img, gt, mask = self.im_stack[hash_im]

            # plot_images([np.squeeze(img)], title="mask")  # 绘制图像
            # plot_images([np.squeeze(img), 255 * np.squeeze(gt)], title="img")  # 绘制图像

            mask = (mask > 0).astype(int)  # binarize the ground-truth
            gt = (gt > 0).astype(int)  # binarize the ground-truth

            patch = img[x_p:x_p + self.p_shape[0], y_p:y_p + self.p_shape[1]]
            label = gt[x_p:x_p + self.p_shape[0], y_p:y_p + self.p_shape[1]]
            ma = mask[x_p:x_p + self.p_shape[0], y_p:y_p + self.p_shape[1]]

            # plot_images([np.squeeze(patch), 255 * np.squeeze(ma)], title="img")  # 绘制图像
            # plot_images([np.squeeze(mask), 255 * np.squeeze(mask)], title="patch")  # 绘制图像

            place_im = img.copy()
            place_im[x_p:x_p + self.p_shape[0], y_p:y_p + self.p_shape[1]] = 10
            place_label = gt.copy()
            place_label[x_p:x_p + self.p_shape[0], y_p:y_p + self.p_shape[1]] = 10

            batch_x.append(patch)
            batch_y.append(label)
            batch_z.append(ma)

            batch_place.append([place_im, place_label])

        return np.array(batch_x), np.array(batch_y), np.array(batch_z), batch_place

    def on_epoch_end(self, epoch=None, logs=None):
        self.im_stack = {}


def get_gen(dataset_ids, p_shape, batch_size=1, gamma=0.9,
            clahe=True, gray=False, xyz=False, hed=False,
            width_shift_range=0, height_shift_range=0,
            horizontal_flip=False, vertical_flip=False,
            rotate_range=0, resize=None,
            MIN_PATCH_STD=None, MAX_IM_QUEUE=100):
    prepro_cfg = dict(gamma=1, horizontal_flip=horizontal_flip,
                      vertical_flip=vertical_flip, width_shift_range=width_shift_range,
                      height_shift_range=height_shift_range, clahe=True, gray=gray, xyz=xyz, hed=hed, green=True)

    prepro_fn = functools.partial(_process_imgt, **prepro_cfg)  # 图像增强策略
    reader_cfg = dict(resize=resize)
    reader_fn = functools.partial(_process_pathnames, **reader_cfg)  # 图像的路径

    return Patch_Sequence(dataset_ids, p_shape=p_shape,
                          reader_fn=reader_fn, preproc_fn=prepro_fn,
                          batch_size=batch_size, MAX_IM_QUEUE=MAX_IM_QUEUE, resize=image_shape[:2])


# file_names = './data\DRIVE\DRIVEtesting.txt'
# file_names = './data\DRIVE\DRIVEtraining.txt'
# database_dir = './data\DRIVE\\'

# file_names = './data\CHASEDB1\\CHASEDB1testing.txt'
# file_names = './data\CHASEDB1\\CHASEDB1training.txt'
# database_dir = './data\CHASEDB1'

# file_names = './data\STARE\\StareTestingFold2.txt'
file_names = './data\STARE\\StareTrainingFold2.txt'
database_dir = './data\Stare'

# save_patch_dir = './patch_result_train\images\\'
# save_label_dir = './patch_result_train\labels\\'
# save_mask_dir = './patch_result_train\masks\\'

save_patch_dir = './patch_result_test\images\\'
save_label_dir = './patch_result_test\labels\\'
save_mask_dir = './patch_result_test\masks\\'

gray = False
xyz = False
clahe = True
gamma = 1

# image_shape = (576, 576, 3)# DRIVE
# image_shape = (960,960,3) #CHASEDB1
image_shape = (672, 672, 3)  # 96-STARE专用

p_shape = (96, 96, 1)  # 分成多少patch

p_w = p_shape[0]
p_h = p_shape[1]


def get_patch():
    X, Y, Z = read_df(file_names, database_dir)

    n = len(X)
    for i in range(0, n):
        patches_positions = fixed_patch_ids_creation([X[i]], [Y[i]], [Z[i]], spatial_shape=image_shape[:2], p_stride=64,
                                                     shuffle=False, )

        # patches_positions = fixed_patch_ids_creation(X, Y, Z, spatial_shape=image_shape[:2], p_stride=16,
        # shuffle=True,) 这里面的patch_gen是所有图片的patch值

        patch_gen = get_gen(patches_positions, p_shape, batch_size=1,
                            gamma=gamma, horizontal_flip=0, width_shift_range=0,
                            height_shift_range=0, vertical_flip=0, rotate_range=0,
                            clahe=clahe, gray=gray, resize=image_shape[:2])

        # 拿到图片的名字，方便保存
        tmp = X[i]
        # DRIVE
        # image_name = tmp[54:56]
        # CHASED
        # image_name = tmp[63:66]
        # Stare
        image_name = tmp[56:60]

        if not os.path.exists(r'%s%s' % (save_patch_dir, image_name)):
            os.makedirs(r'%s%s' % (save_patch_dir, image_name))
        os.path.join(save_patch_dir, image_name)

        if not os.path.exists(r'%s\\%s' % (save_label_dir, image_name)):
            os.makedirs(r'%s\\%s' % (save_label_dir, image_name))
        os.path.join(save_label_dir, image_name)

        if not os.path.exists(r'%s\\%s' % (save_mask_dir, image_name)):
            os.makedirs(r'%s\\%s' % (save_mask_dir, image_name))
        os.path.join(save_mask_dir, image_name)

        xxxx = len(patch_gen)
        for i in range(0, xxxx):
            batch_patch, batch_target, batch_mask, batch_places = patch_gen[i]

            _, h, w, _ = batch_patch.shape

            if h != p_h or w != p_w:
                i = i - 1
                print(i)
                continue

            # plot_images([np.squeeze(batch_patch), np.squeeze(batch_patch)], title="patch")  # 绘制图像

            # image_name_tmp = str(i+1).zfill(5) + 'training.png'
            image_name_tmp = image_name + '/' + str(i + 1).zfill(5) + 'test.png'

            result_patch_path = os.path.join(save_patch_dir, image_name_tmp)
            result_label_path = os.path.join(save_label_dir, image_name_tmp)
            result_mask_path = os.path.join(save_mask_dir, image_name_tmp)

            batch_target = batch_target.astype(int)
            batch_mask = batch_mask.astype(int)

            print(save_label_dir)

            io.imsave(result_patch_path, np.squeeze(batch_patch))
            io.imsave(result_label_path, 255 * np.squeeze(batch_target))
            io.imsave(result_mask_path, 255 * np.squeeze(batch_mask))


if __name__ == '__main__':
    get_patch()
