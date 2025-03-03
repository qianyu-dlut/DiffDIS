import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import numpy as np
import torch
from PIL import ImageEnhance
from utils.image_util import resize_max_res
import cv2
from torch.utils.data.dataset import ConcatDataset



# several data augumentation strategies
def cv_random_flip(img, label,mask):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, mask


def randomCrop(image, label, mask):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), mask.crop(random_region)


def randomRotation(image, label, mask):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        mask = mask.rotate(random_angle, mode)

    return image, label, mask


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    return mask



class DISDataset_wcontour_cutmix(data.Dataset):
    def __init__(self, image_root, gt_root, mask_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('tif')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('tif')]
        self.masks = [mask_root + f for f in os.listdir(mask_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('tif')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)

        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.resize = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize))])
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        image, gt, mask = cv_random_flip(image, gt, mask)
        image, gt, mask = randomCrop(image, gt, mask)
        image, gt, mask = randomRotation(image, gt, mask)
        image = colorEnhance(image)
        image = self.resize(image)
        gt = self.resize(gt)
        mask = self.resize(mask)

        image = np.array(image)
        image = np.transpose(image,(2,0,1))
        gt = np.array(gt)
        mask = np.array(mask)        

        box = obtain_cutmix_box(image.shape[-1])
        image = (((image - image.min()) / (image.max() - image.min())) * 2) - 1
        gt = (((gt - gt.min()) / (gt.max() - gt.min())) * 2) - 1
        mask = (((mask - mask.min()) / (mask.max() - mask.min())) * 2) - 1

        image = torch.from_numpy(image)
        gt = torch.from_numpy(gt)
        mask = torch.from_numpy(mask)
        return image, gt, mask, box 

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        masks = []

        for img_path, gt_path, mask_path in zip(self.images, self.gts, self.masks):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            mask = Image.open(mask_path)

            if img.size == gt.size :
                images.append(img_path)
                gts.append(gt_path)
                masks.append(mask_path)

        self.images = images
        self.gts = gts
        self.masks = masks


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):
    dataset = DISDataset_wcontour_cutmix(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=True)
    return data_loader




 