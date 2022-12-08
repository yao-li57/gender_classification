import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


def read_split_data(imgs, labels, val_ratio=0.2, plot=False):
    # random.seed(0)
    assert os.path.exists(imgs), "imgs root:{} does not exist".format(imgs)
    assert os.path.exists(labels), "imgs root:{} does not exist".format(labels)

    train_imgs_path = []
    train_labels = []
    val_imgs_path = []
    val_labels = []
    supported = ['.jpg', '.JPG']

    # 得到所有图片地址
    images_name = os.listdir(imgs)
    images = [os.path.join(imgs, img) for img in os.listdir(imgs)
              if os.path.splitext(img)[-1] in supported]
    # 得到所有图片label
    df = pd.read_csv(labels)
    path_label_dict = {}
    for i in images_name:
        path_label_dict.update({os.path.join(imgs, i): df[df.image_id == i].iloc[0, 1]})

    # 按比例划分val
    val_num = len(images) * val_ratio
    val_path = random.sample(images, int(val_num))

    for img_path in images:
        if img_path in val_path:
            val_imgs_path.append(img_path)
            val_labels.append(path_label_dict[img_path])
        else:
            train_imgs_path.append(img_path)
            train_labels.append(path_label_dict[img_path])

    print("{} images were found in the dataset".format(len(images)))
    print("{} in train_set".format(len(train_labels)))
    print("{} in val_set".format(len(val_labels)))

    if plot:
        train_male_num = np.sum(np.array(train_labels) == 1)
        train_female_num = np.sum(np.array(train_labels) == 0)
        val_male_num = np.sum(np.array(val_labels) == 1)
        val_female_num = np.sum(np.array(val_labels) == 0)
        x = ['train_m', 'train_f', 'val_m', 'val_f']
        y = [train_male_num, train_female_num, val_male_num, val_female_num]
        plt.bar(x, y, color=['b', 'r', 'b', 'r'])
        for a, b, i in zip(x, y, range(len(x))):  # zip 函数
            plt.text(a, b, "%d" % y[i], ha='center', fontsize=12)  # plt.text 函数
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
        plt.title("数据集性别分布")
        plt.show()

    return train_imgs_path, train_labels, val_imgs_path, val_labels

class MyDataset(Dataset):
    def __init__(self, image_path: list, image_class: list, transform=None):
        self.image_path = image_path
        self.image_class = image_class
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        img = Image.open(self.image_path[item]).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode".format(self.image_path[item]))
        label = self.image_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
