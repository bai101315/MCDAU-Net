import os.path
from PIL import Image
import sys
import torchvision.transforms as transforms
import cv2
import numpy as np
from image_test import test


# 图片的有间隙拼接
def image_compose(IMAGE_SIZE, IMAGE_ROW, IMAGE_COLUMN, padding, IMAGES_PATH,IMAGE_SAVE_PATH):
    IMAGES_FORMAT = ['.bmp', '.jpg', '.tif', '.png']  # 图片格式
    # 获取图片集地址下的所有图片名称
    image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]

    # 排序，这里需要根据自己的图片名称切割，得到数字
    image_names.sort(key=lambda x: int(x.split(("t"), 2)[0]))
    # 简单的对于参数的设定和实际图片集的大小进行数量判断
    if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
        print(IMAGES_PATH)
        raise ValueError("合成图片的参数和要求的数量不能匹配！")

    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE + padding * (IMAGE_COLUMN-1), IMAGE_ROW * IMAGE_SIZE + padding * (IMAGE_ROW-1)), 'white')  # 创建一个新图,颜色为白色
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    print(images_save_path)

    for y in range(1, IMAGE_ROW + 1 ):
        for x in range(1, IMAGE_COLUMN +1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

            # from_image = np.array(from_image)
            # from_image = from_image[16:112,16:112]
            #
            # from_image=Image.fromarray(from_image) # numpy 转 image类

            # to_image.paste(from_image, (
            # (x - 1) * IMAGE_SIZE + padding * (x - 1),  (y - 1) * IMAGE_SIZE + padding * (y - 1)))
            to_image.paste(from_image, (
            (x - 1) * IMAGE_SIZE + padding * (x - 1),  (y - 1) * IMAGE_SIZE + padding * (y - 1)))

    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图



if __name__ == '__main__':
    padding = 0

    # DRIVE
    # IMAGE_SIZE = 64  # 每张小图片的大小
    # IMAGE_ROW = 9 # 图片间隔，也就是合并成一张图后，一共有几行
    # IMAGE_COLUMN = 9  # 图片间隔，也就是合并成一张图后，一共有几列

    # # CHASEDB1
    # IMAGE_SIZE = 64  # 每张小图片的大小
    # IMAGE_ROW = 15 # 图片间隔，也就是合并成一张图后，一共有几行
    # IMAGE_COLUMN = 15  # 图片间隔，也就是合并成一张图后，一共有几列

    # SATRE
    IMAGE_SIZE = 64 # 每张小图片的大小
    IMAGE_ROW = 10  # 图片间隔，也就是合并成一张图后，一共有几行
    IMAGE_COLUMN = 10  # 图片间隔，也就是合并成一张图后，一共有几列

    IMAGES_PATH = './result'
    IMAGE_SAVE_PATH = './test/result'
    # BIG_IMAGES_PATH = 'C:\\Users\BAI\Desktop\data\DRIVE_result\96_result\\result'

    image_class = [cla for cla in os.listdir(IMAGES_PATH) if os.path.isdir(os.path.join(IMAGES_PATH, cla))]
    # big_image_class  = [cla for cla in os.listdir(BIG_IMAGES_PATH) if os.path.isdir(os.path.join(BIG_IMAGES_PATH, cla))]

    Background_IOU = []
    Vessel_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []

    # DRIVE
    for i in image_class:

        if i == "result" or i == "test":
            continue

        im_id=str(i).zfill(2)
        images_path = os.path.join(IMAGES_PATH,im_id + '//')


        images_save_path = os.path.join(IMAGE_SAVE_PATH,im_id+'_test.tif')

        image_compose(IMAGE_SIZE, IMAGE_ROW, IMAGE_COLUMN, padding, images_path,images_save_path)  # 调用函数

    test()




