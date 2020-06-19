import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms

import faceutils as futils

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


def to_var(x, requires_grad=True):
    if requires_grad:
        return Variable(x).float()
    else:
        return Variable(x, requires_grad=requires_grad).float()


def copy_area(tar, src, lms):
    rect = [int(min(lms[:, 1])) - preprocess_image.eye_margin,
            int(min(lms[:, 0])) - preprocess_image.eye_margin,
            int(max(lms[:, 1])) + preprocess_image.eye_margin + 1,
            int(max(lms[:, 0])) + preprocess_image.eye_margin + 1]
    tar[:, :, rect[1]:rect[3], rect[0]:rect[2]] = \
        src[:, :, rect[1]:rect[3], rect[0]:rect[2]]


def preprocess_image(image: Image):
    face = futils.dlib.detect(image)

    assert face, "no faces detected"

    # face[0]是第一个人脸，给定图片中只能有一个人脸
    face = face[0]
    image, face = futils.dlib.crop(image, face)

    # detect landmark
    lms = futils.dlib.landmarks(image, face) * 256 / image.width
    lms = lms.round()
    lms_eye_left = lms[42:48]
    lms_eye_right = lms[36:42]
    lms = lms.transpose((1, 0)).reshape(-1, 1, 1)  # transpose to (y-x)
    lms = np.tile(lms, (1, 256, 256))  # (136, h, w)

    # calculate relative position for each pixel
    fix = np.zeros((256, 256, 68 * 2))
    for i in range(256):  # row (y) h
        for j in range(256):  # column (x) w
            fix[i, j, :68] = i
            fix[i, j, 68:] = j
    fix = fix.transpose((2, 0, 1))  # (136, h, w)
    diff = to_var(torch.Tensor(fix - lms).unsqueeze(0), requires_grad=False)

    # obtain face parsing result
    image = image.resize((512, 512), Image.ANTIALIAS)
    mask = futils.mask.mask(image).resize((256, 256), Image.ANTIALIAS)
    mask = to_var(ToTensor(mask).unsqueeze(0), requires_grad=False)
    mask_lip = (mask == 7).float() + (mask == 9).float()
    mask_face = (mask == 1).float() + (mask == 6).float()

    # 需要抠出 mask_eye
    mask_eyes = torch.zeros_like(mask)
    copy_area(mask_eyes, mask_face, lms_eye_left)
    copy_area(mask_eyes, mask_face, lms_eye_right)
    mask_eyes = to_var(mask_eyes, requires_grad=False)

    mask_list = [mask_lip, mask_face, mask_eyes]
    mask_aug = torch.cat(mask_list, 0)  # (3, 1, h, w)
    # print('mask_aug shape: ', mask_aug.shape)
    # 根据给定 size 或 scale_factor，上采样或下采样输入数据input
    mask_re = F.interpolate(mask_aug, size=preprocess_image.diff_size).repeat(1, diff.shape[1], 1, 1)  # (3, 136, 64, 64)
    diff_re = F.interpolate(diff, size=preprocess_image.diff_size).repeat(3, 1, 1, 1)  # (3, 136, 64, 64)
    # 这就是论文里计算attention时要求同一个facial region
    diff_re = diff_re * mask_re  # (3, 136, 64, 64)
    # dim=1，求出的norm就是(3, 1, 64, 64)，也就是relative position的范数值
    norm = torch.norm(diff_re, dim=1, keepdim=True).repeat(1, diff_re.shape[1], 1, 1)
    # torch.where()函数的作用是按照一定的规则合并两个tensor类型
    norm = torch.where(norm == 0, torch.tensor(1e10), norm)
    diff_re /= norm

    image = image.resize((256, 256), Image.ANTIALIAS)
    real = to_var(transform(image).unsqueeze(0))
    # print('real shape: ', real.shape)
    # return [real, mask_aug, diff_re]
    return [real, mask_list, diff_re, mask_aug]

# parameter of eye transfer
preprocess_image.eye_margin = 16
# down sample size
preprocess_image.diff_size = (64, 64)

class MAKEUP(Dataset):
    def __init__(self, image_path, transform, mode, transform_mask, cls_list):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.transform_mask = transform_mask

        self.cls_list = cls_list
        self.cls_A = cls_list[0]
        self.cls_B = cls_list[1]

        for cls in self.cls_list:
            setattr(self, "train_" + cls + "_list_path", os.path.join(self.image_path, "train_" + cls + ".txt"))
            setattr(self, "train_" + cls + "_lines",
                    open(getattr(self, "train_" + cls + "_list_path"), 'r').readlines())
            setattr(self, "num_of_train_" + cls + "_data", len(getattr(self, "train_" + cls + "_lines")))
        for cls in self.cls_list:
            if self.mode == "test_all":
                setattr(self, "test_" + cls + "_list_path", os.path.join(self.image_path, "test_" + cls + "_all.txt"))
                setattr(self, "test_" + cls + "_lines",
                        open(getattr(self, "test_" + cls + "_list_path"), 'r').readlines())
                setattr(self, "num_of_test_" + cls + "_data", len(getattr(self, "test_" + cls + "_lines")))
            else:
                setattr(self, "test_" + cls + "_list_path", os.path.join(self.image_path, "test_" + cls + ".txt"))
                setattr(self, "test_" + cls + "_lines",
                        open(getattr(self, "test_" + cls + "_list_path"), 'r').readlines())
                setattr(self, "num_of_test_" + cls + "_data", len(getattr(self, "test_" + cls + "_lines")))

        print('Start preprocessing dataset..!')
        self.preprocess_dataset()
        print('Finished preprocessing dataset..!')

    def preprocess_dataset(self):
        for cls in self.cls_list:
            setattr(self, "train_" + cls + "_filenames", [])
            setattr(self, "train_" + cls + "_mask_filenames", [])

            lines = getattr(self, "train_" + cls + "_lines")
            random.shuffle(lines)

            # 所以生成的txt文件每行两个字符串，一个是image，一个是mask
            for i, line in enumerate(lines):
                splits = line.split()
                getattr(self, "train_" + cls + "_filenames").append(splits[0])
                getattr(self, "train_" + cls + "_mask_filenames").append(splits[1])

        for cls in self.cls_list:
            setattr(self, "test_" + cls + "_filenames", [])
            setattr(self, "test_" + cls + "_mask_filenames", [])
            lines = getattr(self, "test_" + cls + "_lines")
            for i, line in enumerate(lines):
                splits = line.split()
                getattr(self, "test_" + cls + "_filenames").append(splits[0])
                getattr(self, "test_" + cls + "_mask_filenames").append(splits[1])

        if self.mode == "test_baseline":
            setattr(self, "test_" + self.cls_A + "_filenames",
                    os.listdir(os.path.join(self.image_path, "baseline", "org_aligned")))
            setattr(self, "num_of_test_" + self.cls_A + "_data",
                    len(os.listdir(os.path.join(self.image_path, "baseline", "org_aligned"))))
            setattr(self, "test_" + self.cls_B + "_filenames",
                    os.listdir(os.path.join(self.image_path, "baseline", "ref_aligned")))
            setattr(self, "num_of_test_" + self.cls_B + "_data",
                    len(os.listdir(os.path.join(self.image_path, "baseline", "ref_aligned"))))

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'train_finetune':
            index_A = random.randint(0, getattr(self, "num_of_train_" + self.cls_A + "_data") - 1)
            index_B = random.randint(0, getattr(self, "num_of_train_" + self.cls_B + "_data") - 1)
            # 这里image_path是通过./data+txt中的image_path得到的，所以txt中应该写的是./data的相对地址，比如/makeup/xxx
            image_A = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_A + "_filenames")[index_A]))
            image_B = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_B + "_filenames")[index_B]))

            image_A_list = preprocess_image(image_A)
            image_B_list = preprocess_image(image_B)

            return image_A_list, image_B_list
            # return self.transform(image_A), self.transform(image_B)
        if self.mode in ['test', 'test_all']:
            # """
            image_A = Image.open(os.path.join(self.image_path, getattr(self, "test_" + self.cls_A + "_filenames")[
                index // getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')]))
            image_B = Image.open(os.path.join(self.image_path, getattr(self, "test_" + self.cls_B + "_filenames")[
                index % getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')]))

            image_A_list = preprocess_image(image_A)
            image_B_list = preprocess_image(image_B)
            return image_A_list, image_B_list
        if self.mode == "test_baseline":
            image_A = Image.open(os.path.join(self.image_path, "baseline", "org_aligned",
                                              getattr(self, "test_" + self.cls_A + "_filenames")[index // getattr(self,
                                                                                                                  'num_of_test_' +
                                                                                                                  self.cls_list[
                                                                                                                      1] + '_data')])).convert(
                "RGB")
            image_B = Image.open(os.path.join(self.image_path, "baseline", "ref_aligned",
                                              getattr(self, "test_" + self.cls_B + "_filenames")[index % getattr(self,
                                                                                                                 'num_of_test_' +
                                                                                                                 self.cls_list[
                                                                                                                     1] + '_data')])).convert(
                "RGB")
            return self.transform(image_A), self.transform(image_B)

    def __len__(self):
        # 为啥train就是max(A,B)，test就是相乘呢?
        # 这里应该是为了测试的时候实验同一张source对应不同reference的结果
        if self.mode == 'train' or self.mode == 'train_finetune':
            num_A = getattr(self, 'num_of_train_' + self.cls_list[0] + '_data')
            num_B = getattr(self, 'num_of_train_' + self.cls_list[1] + '_data')
            return max(num_A, num_B)
        elif self.mode in ['test', "test_baseline", 'test_all']:
            num_A = getattr(self, 'num_of_test_' + self.cls_list[0] + '_data')
            num_B = getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')
            return num_A * num_B
