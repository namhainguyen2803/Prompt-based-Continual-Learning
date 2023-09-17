import os
import os.path
import hashlib
import errno
from six.moves import urllib
import torch
from torchvision import transforms
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torchvision import transforms as T
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

class ColorJitter:
    def __init__(
        self,
        p,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue= 0.0,
    ):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        if np.random.uniform(0,1) > self.p:
            return img
        return transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(img)

dataset_stats = {
    'CIFAR10': {'mean': (0.49139968, 0.48215827 ,0.44653124),
                 'std' : (0.24703233, 0.24348505, 0.26158768),
                 'size' : 32},
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32},
    'ImageNet_R': {
                 'size' : 224}, 
                }

# transformations
def get_transform(dataset='cifar100', phase='test', aug=True, resize_imnet=False):
    transform_list = []
    # get out size
    crop_size = dataset_stats[dataset]['size']

    # get mean and std
    dset_mean = (0.0,0.0,0.0) # dataset_stats[dataset]['mean']
    dset_std = (1.0,1.0,1.0) # dataset_stats[dataset]['std']

    if dataset == "CIFAR100":
        if phase == 'train':
            transform_list.extend([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.1),
                transforms.RandomSolarize(0.2),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std)
            ])
        else:
            transform_list.extend([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std)
            ])
    else:
        if dataset == 'ImageNet32' or dataset == 'ImageNet84':
            transform_list.extend([
                transforms.Resize((crop_size,crop_size))
            ])

        if phase == 'train':
            transform_list.extend([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
        else:
            if dataset.startswith('ImageNet') or dataset == 'DomainNet':
                transform_list.extend([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(dset_mean, dset_std),
                                    ])
            else:
                transform_list.extend([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(dset_mean, dset_std),
                                    ])


    return transforms.Compose(transform_list)

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)