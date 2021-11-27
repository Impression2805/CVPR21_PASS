import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets, transforms
import os
import sys
import pdb
import random


class tinyImageNet(object):
    def download_data(self):
        train_dir = os.path.join('../dataset', 'tiny-imagenet', 'tiny-imagenet-200', 'train')
        test_dir = os.path.join('../dataset', 'tiny-imagenet', 'tiny-imagenet-200', 'val')
        train_dset = datasets.ImageFolder(train_dir)

        train_images = []
        train_labels = []
        for item in train_dset.imgs:
            train_images.append(item[0])
            train_labels.append(item[1])
        self.train_data, self.train_targets = np.array(train_images), np.array(train_labels)

        test_images = []
        test_labels = []
        _, class_to_idx = find_classes(train_dir)
        imgs_path = os.path.join(test_dir, 'images')
        imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())
        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        for imgname in sorted(os.listdir(imgs_path)):
            if cls_map[imgname] in sorted(class_to_idx.keys()):
                path = os.path.join(imgs_path, imgname)
                test_images.append(path)
                test_labels.append(class_to_idx[cls_map[imgname]])
        self.test_data, self.test_targets = np.array(test_images), np.array(test_labels)


class DataManager(object):
    def __init__(self):
        self._setup_data()

    def get_dataset(self, transform, index, train=True):
        if train:
            x, y = self._train_data, self._train_targets
        else:
            x, y = self._test_data, self._test_targets

        data, targets = [], []
        for idx in index:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)
        return DummyDataset(data, targets, transform)

    def _setup_data(self, shuffle=True, seed=1993):
        idata = tinyImageNet()
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = np.arange(200).tolist()
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(pil_loader(self.images[idx]))
        label = self.labels[idx]

        return image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
