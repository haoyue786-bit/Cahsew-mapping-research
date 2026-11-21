from collections import Counter
import os
from osgeo import gdal
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
import cv2
from .config2 import *

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def preprocess_input(image):
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)
    return image


# NOTE: cleaned English comment.
def create_patches_train_sequential(img_array, label_array, patchSize, switch_labelValid=False):
    height, width = label_array.shape
    image_patches = []
    label_patches = []
    count = 0
    output_patch_size = patchSize - patchSize//4
    stride = patchSize // 2 # NOTE: cleaned comment.
    borderRemoved = patchSize//8
    for i in range(height // stride):
        for j in range(width // stride):

            i_label_start, i_label_end = i * stride+borderRemoved, i * stride + borderRemoved + output_patch_size
            i_image_start, i_image_end = i * stride, i * stride + patchSize
            j_label_start, j_label_end = j * stride+borderRemoved, j * stride + borderRemoved + output_patch_size
            j_image_start, j_image_end = j * stride, j * stride + patchSize
            if 0 <= i_image_start < height and 0 < i_image_end <= height and 0 <= j_image_start < width and 0 < j_image_end <= width:
                count += 1
                image_patch = img_array[:, :, i_image_start:i_image_end, j_image_start:j_image_end]
                label_patch = label_array[i_label_start:i_label_end, j_label_start:j_label_end]

                image_patches.append(image_patch)
                label_patches.append(label_patch)

    print(f"The number of clipped patches: {count}")
    return image_patches, label_patches

def create_patches_train_sequential_dispersed(img_array, label_array, label_array_valid, patchSize, threshold=0.5):
    height, width = label_array.shape
    image_patches = []
    label_patches = []
    count = 0
    output_patch_size = patchSize - patchSize//4
    stride = patchSize // 4
    borderRemoved = patchSize//8
    for i in range(height // stride):
        for j in range(width // stride):

            i_label_start, i_label_end = i * stride+borderRemoved, i * stride + borderRemoved + output_patch_size
            i_image_start, i_image_end = i * stride, i * stride + patchSize
            j_label_start, j_label_end = j * stride+borderRemoved, j * stride + borderRemoved + output_patch_size
            j_image_start, j_image_end = j * stride, j * stride + patchSize
            if 0 <= i_image_start < height and 0 < i_image_end <= height and 0 <= j_image_start < width and 0 < j_image_end <= width:

                image_patch = img_array[:, :, i_image_start:i_image_end, j_image_start:j_image_end]
                label_patch = label_array[i_label_start:i_label_end, j_label_start:j_label_end]
                label_valid_patch = label_array_valid[i_label_start:i_label_end, j_label_start:j_label_end]

                if np.sum(label_valid_patch) >= threshold * output_patch_size * output_patch_size:
                    count += 1
                    image_patches.append(image_patch)
                    label_patches.append(label_patch)

    print(f"The number of clipped patches: {count}")
    return image_patches, label_patches

class dataset(Dataset):

    def __init__(self, image_patches, label_patches, transform=None):
        self.image_patches = image_patches
        self.label_patches = label_patches
        self.transform = transform

    def __len__(self):
        return len(self.label_patches)

    def __getitem__(self, index):
        if self.transform:
            image_patch = self.transform(self.image_patches[index])

        return self.image_patches[index], self.label_patches[index]



DA = True  # NOTE: cleaned comment.



class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, annotation_lines_tgr, input_shape, num_classes, time_step, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.DA = DA
        self.annotation_lines = annotation_lines
        self.annotation_lines_tgr = annotation_lines_tgr
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.time_dim = len(time_step)  # NOTE: cleaned comment.

        self.img_source = os.path.join(dataset_path, "split_train_sentinel")
        self.img_target = os.path.join(dataset_path, "split_train_DA")
        # NOTE: cleaned English comment.
        self.label_dir = os.path.join(dataset_path, "split_train_label")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # NOTE: cleaned English comment.
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]
        # NOTE: cleaned English comment.
        jpg_imgs = []  # NOTE: cleaned comment.
        tgr_imgs = []
        for time in self.time_step:
            # NOTE: cleaned English comment.
            img_path = os.path.join(self.img_source, str(time), f"{name}")
            # NOTE: cleaned English comment.
            jpg = gdal.Open(img_path)

            if jpg is None:
                raise FileNotFoundError(f"Image file not found: {img_path}")

            # NOTE: cleaned English comment.
            jpg_im_width = jpg.RasterXSize
            jpg_im_height = jpg.RasterYSize
            jpg_data = jpg.ReadAsArray(0, 0, jpg_im_width, jpg_im_height)  # NOTE: cleaned comment.
            del jpg  # NOTE: cleaned comment.

            # NOTE: cleaned English comment.
            if len(jpg_data.shape) == 2:
                jpg_data = np.expand_dims(jpg_data, axis=0)  # NOTE: cleaned comment.

            jpg_imgs.append(jpg_data)

        if self.DA and self.train:
            annotation_line_tgr = self.annotation_lines_tgr[index]
            name_DA = annotation_line_tgr.split()[0]
            for time in self.time_step:
                tgr_path = os.path.join(self.img_target, str(time), f"{name_DA}")
                tgr = gdal.Open(tgr_path)

                if tgr is None:
                    raise FileNotFoundError(f"Target image file not found: {tgr_imgs}")

                tgr_data = tgr.ReadAsArray(0, 0, jpg_im_width, jpg_im_height)  # NOTE: cleaned comment.
                del tgr  # NOTE: cleaned comment.

                if len(tgr_data.shape) == 2:
                    tgr_data = np.expand_dims(tgr_data, axis=0)

                tgr_imgs.append(tgr_data)

        # NOTE: cleaned English comment.
        # NOTE: cleaned English comment.

        # NOTE: cleaned English comment.
        seg_path = os.path.join(self.label_dir, f"{name}")
        seg = gdal.Open(seg_path)
        if seg is None:
            raise FileNotFoundError(f"Label file not found: {seg_path}")

        seg_png = seg.ReadAsArray(0, 0, seg.RasterXSize, seg.RasterYSize)
        del seg
        # ==========================================
        # NOTE: cleaned English comment.

        if self.train:
            img_t0 = jpg_imgs[0]  # NOTE: cleaned comment.
            img_t1 = jpg_imgs[1]  # NOTE: cleaned comment.
            img_t0, img_t1, seg_png = self.get_random_data(
                img_t0, img_t1, seg_png, self.input_shape
            )
            jpg_imgs_aug = []
            jpg_imgs_aug.append(img_t0)
            jpg_imgs_aug.append(img_t1)
            stacked_imgs = np.stack(jpg_imgs_aug, axis=0)
        else:
            stacked_imgs = np.stack(jpg_imgs, axis=0)
        # ==========================================
        # seg_png = np.clip(seg_png, 0, self.num_classes - 1)


        if self.DA and self.train:
            stacked_imgs_src = np.stack(jpg_imgs, axis=0)
            stacked_imgs_tgr = np.stack(tgr_imgs, axis=0)
            min_samples = min(len(stacked_imgs_src), len(stacked_imgs_tgr))
            stacked_imgs_src = stacked_imgs_src[:min_samples]  # NOTE: cleaned comment.
            stacked_imgs_tgr = stacked_imgs_tgr[:min_samples]  # NOTE: cleaned comment.
            seg_png_min = seg_png[:min_samples]

            all_images = np.concatenate([stacked_imgs_src, stacked_imgs_tgr], axis=0)
            label_patches_src_DA_train = 0 * np.ones(seg_png_min.shape).astype(np.uint8)
            label_patches_tgt_DA_train = 1 * np.ones(seg_png_min.shape).astype(np.uint8)
            all_domain_labels = np.concatenate([label_patches_src_DA_train, label_patches_tgt_DA_train], axis=0)

            num_samples = all_images.shape[0]  # NOTE: cleaned comment.
            shuffled_indices = np.random.permutation(num_samples)  # NOTE: cleaned comment.

            # NOTE: cleaned English comment.
            shuffled_images = all_images[shuffled_indices]
            shuffled_domain_labels = all_domain_labels[shuffled_indices]


            DA_train_images_patches = shuffled_images[:min_samples]  # NOTE: cleaned comment.
            DA_train_labels_patches = shuffled_domain_labels[:min_samples]
            # NOTE: cleaned English comment.
            # print("seg_png",seg_png.shape)
            # print("DA_train_labels_patches", DA_train_labels_patches.shape)
            return stacked_imgs, seg_png, DA_train_images_patches, DA_train_labels_patches
        else:
            return stacked_imgs, seg_png
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a


    def get_random_data(self, img_t0, img_t1, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3):

        img_t0 = np.transpose(img_t0.astype(np.uint8),(1, 2, 0))  # NOTE: cleaned comment.
        img_t1 = np.transpose(img_t1.astype(np.uint8),(1, 2, 0))  # NOTE: cleaned comment.
        img_t0 = Image.fromarray(img_t0)  # NOTE: cleaned comment.
        img_t1 = Image.fromarray(img_t1)

        label = Image.fromarray(label.astype(np.uint8))  # NOTE: cleaned comment.

        # ------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------#
        iw, ih = img_t0.size
        h, w = input_shape
        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(1, 2)

        nw = int(scale * w)
        nh = int(nw / new_ar)
        img_t0 = img_t0.resize((nw, nh), Image.BICUBIC)
        img_t1 = img_t1.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            img_t0 = img_t0.transpose(Image.FLIP_LEFT_RIGHT)
            img_t1 = img_t1.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))

        # NOTE: cleaned English comment.
        new_image_t0 = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        new_image_t1 = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        new_image_t0.paste(img_t0, (dx, dy))  # NOTE: cleaned comment.
        new_image_t1.paste(img_t1, (dx, dy))

        image_t0 = new_image_t0
        image_t1 = new_image_t1
        new_label = Image.new('L', (w, h), (0))
        new_label.paste(label, (dx, dy))
        label = new_label

        image_data_t0 = np.array(image_t0, np.uint8)
        image_data_t1 = np.array(image_t1, np.uint8)
        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        blur = self.rand() < 0.25
        if blur:
            image_data_t0 = cv2.GaussianBlur(image_data_t0, (5, 5), 0)
            image_data_t1 = cv2.GaussianBlur(image_data_t1, (5, 5), 0)

        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            # image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))

            image_data_t0 = cv2.warpAffine(
                image_data_t0, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderValue=(0, 0, 0, 0)  # NOTE: cleaned comment.
            )
            image_data_t1 = cv2.warpAffine(
                image_data_t1, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderValue=(0, 0, 0, 0)
            )
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        # ---------------------------------#
        # NOTE: cleaned English comment.
        # NOTE: cleaned English comment.
        # ---------------------------------#


        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data_t0, cv2.COLOR_RGB2HSV))
        dtype = image_data_t0.dtype
            # # ---------------------------------#
            # NOTE: cleaned English comment.
            # # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        img_t0 = np.transpose(image_data_t0, (2, 0, 1))
        img_t1 = np.transpose(image_data_t1, (2, 0, 1))

        img_t0 = np.array(img_t0)  # NOTE: cleaned comment.
        img_t1 = np.array(img_t1)
        label = np.array(label)
        return img_t0,img_t1,label

class SegmentationDataset_AEF(Dataset):
    def __init__(self, annotation_lines, annotation_lines_tgr, input_shape, num_classes, train,
                 dataset_path):
        super(SegmentationDataset_AEF, self).__init__()
        self.DA = True  # Domain adaptation switch.
        self.annotation_lines = annotation_lines  # Source-domain sample list.
        self.annotation_lines_tgr = annotation_lines_tgr  # Target-domain sample list.
        self.length = len(annotation_lines)
        self.input_shape = input_shape  # Input image shape (h, w).
        self.num_classes = num_classes  # Number of segmentation classes.
        self.train = train  # Training flag (controls augmentation).
        self.dataset_path = dataset_path  # Dataset root path.

        # Source images, target images, and segmentation labels.
        self.img_source = os.path.join(dataset_path, "split_train_sentinel")
        self.img_target = os.path.join(dataset_path, "split_train_DA")
        self.label_dir = os.path.join(dataset_path, "split_train_label")

    def __len__(self):
        if self.DA and self.train:
            return max(len(self.annotation_lines), len(self.annotation_lines_tgr))
        return len(self.annotation_lines)

    def __getitem__(self, index):
        src_index = index % len(self.annotation_lines)
        annotation_line = self.annotation_lines[src_index]
        name = annotation_line.split()[0]

        img_path = os.path.join(self.img_source, f"{name}")
        img = gdal.Open(img_path)
        if img is None:
            raise FileNotFoundError(f"Source image not found: {img_path}")

        img_width = img.RasterXSize
        img_height = img.RasterYSize
        img_data = img.ReadAsArray(0, 0, img_width, img_height)
        del img

        seg_path = os.path.join(self.label_dir, f"{name}")
        seg = gdal.Open(seg_path)
        if seg is None:
            raise FileNotFoundError(f"Label image not found: {seg_path}")

        seg_data = seg.ReadAsArray(0, 0, seg.RasterXSize, seg.RasterYSize)
        del seg

        if self.DA and self.train:
            img_data_aug, seg_data = self.get_random_data(img_data, seg_data)

            tgt_index = index % len(self.annotation_lines_tgr)
            annotation_line_tgr = self.annotation_lines_tgr[tgt_index]
            name_DA = annotation_line_tgr.split()[0]
            tgr_path = os.path.join(self.img_target, f"{name_DA}")
            tgr = gdal.Open(tgr_path)
            if tgr is None:
                raise FileNotFoundError(f"Target-domain image not found: {tgr_path}")
            tgr_data = tgr.ReadAsArray()
            del tgr

            # (src_aug, seg_label, src_raw_for_DA, tgt_raw_for_DA)
            return img_data_aug, seg_data, img_data, tgr_data

        return img_data, seg_data
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, img, label):
        """
        Lightweight augmentation:
        - random horizontal flip
        - random fixed-angle rotation (90/180/270)
        Keeps image dtype as float32 and label dtype as uint8.
        """
        C, H, W = img.shape  # (channels, height, width)

        # ------------------------------#
        # 1) Enforce expected dtypes.
        # ------------------------------#
        img = img.astype(np.float32)
        label = label.astype(np.uint8)

        # ------------------------------#
        # 2) Random horizontal flip.
        # ------------------------------#
        if self.rand() < 0.5:
            for c in range(C):
                img[c, :, :] = cv2.flip(img[c, :, :], 1)
            label = cv2.flip(label, 1)

        # ------------------------------#
        # 3) Random fixed-angle rotation.
        # ------------------------------#
        rotate_choice = np.random.choice([0, 1, 2, 3])  # 0=no rotation, 1/2/3=90/180/270.
        if rotate_choice != 0:
            def rotate_channel(channel, choice):
                if choice == 1:
                    return cv2.rotate(channel, cv2.ROTATE_90_CLOCKWISE)
                elif choice == 2:
                    return cv2.rotate(channel, cv2.ROTATE_180)
                elif choice == 3:
                    return cv2.rotate(channel, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    return channel

            rotated_channels = []
            for c in range(C):
                rotated_channel = rotate_channel(img[c, :, :], rotate_choice)
                rotated_channels.append(rotated_channel)
            img = np.stack(rotated_channels, axis=0)
            label = rotate_channel(label, rotate_choice)

        # ------------------------------#
        # 4) Final dtype normalization.
        # ------------------------------#
        img = img.astype(np.float32)
        label = label.astype(np.uint8)
        return img, label


class SegmentationDataset_AEF_val(Dataset):
    def __init__(self, annotation_lines, annotation_lines_tgr, input_shape, num_classes, train,
                 dataset_path):
        super(SegmentationDataset_AEF_val, self).__init__()
        self.DA = True  # NOTE: cleaned comment.
        self.annotation_lines = annotation_lines  # NOTE: cleaned comment.
        self.annotation_lines_tgr = annotation_lines_tgr  # NOTE: cleaned comment.
        self.length = len(annotation_lines)
        self.input_shape = input_shape  # NOTE: cleaned comment.
        self.num_classes = num_classes  # NOTE: cleaned comment.
        self.train = train  # NOTE: cleaned comment.
        self.dataset_path = dataset_path  # NOTE: cleaned comment.

        # NOTE: cleaned English comment.
        self.img_source = os.path.join(dataset_path, "split_train_DA")#split_train_DA")
        self.label_dir = os.path.join(dataset_path, "0105SWINDA")  # NOTE: cleaned comment.

    def __len__(self):
        return min(len(self.annotation_lines), len(self.annotation_lines_tgr))

    def __getitem__(self, index):
        # NOTE: cleaned English comment.
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]  # NOTE: cleaned comment.

        img_path = os.path.join(self.img_source, f"{name}")
        img = gdal.Open(img_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # NOTE: cleaned English comment.
        img_width = img.RasterXSize
        img_height = img.RasterYSize
        img_data = img.ReadAsArray(0, 0, img_width, img_height)
        del img  # NOTE: cleaned comment.


        # NOTE: cleaned English comment.
        label_name_mapper = getattr(self, "label_name_mapper", None)
        label_name = label_name_mapper(name) if label_name_mapper is not None else name
        seg_path = os.path.join(self.label_dir, f"{label_name}")
        seg = gdal.Open(seg_path)
        if seg is None:
            raise FileNotFoundError(f"Label file not found: {seg_path}")

        seg_data = seg.ReadAsArray(0, 0, seg.RasterXSize, seg.RasterYSize)
        del seg
        # NOTE: cleaned English comment.
        return img_data, seg_data



class SegmentationDataset_AEF2(Dataset):
    def __init__(self, annotation_lines, annotation_lines_tgr, annotation_lines_sor,input_shape, num_classes, train,
                 dataset_path):
        super(SegmentationDataset_AEF2, self).__init__()
        self.DA = True  # NOTE: cleaned comment.
        self.annotation_lines = annotation_lines  # NOTE: cleaned comment.
        self.annotation_lines_sor = annotation_lines_sor
        self.annotation_lines_tgr = annotation_lines_tgr  # NOTE: cleaned comment.
        self.length = len(annotation_lines)
        self.input_shape = input_shape  # NOTE: cleaned comment.
        self.num_classes = num_classes  # NOTE: cleaned comment.
        self.train = train  # NOTE: cleaned comment.
        self.dataset_path = dataset_path  # NOTE: cleaned comment.

        # NOTE: cleaned English comment.
        self.img_source = os.path.join(dataset_path, "split_train_sentinel")  # NOTE: cleaned comment.
        self.img_sourcebig = os.path.join(dataset_path, "split_train_AEF")  # NOTE: cleaned comment.
        self.img_target = os.path.join(dataset_path, "split_train_DA")  # NOTE: cleaned comment.
        self.label_dir = os.path.join(dataset_path, "split_train_label")  # NOTE: cleaned comment.

    def __len__(self):
        return min(len(self.annotation_lines), len(self.annotation_lines_tgr))

    def __getitem__(self, index):
        # NOTE: cleaned English comment.
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]  # NOTE: cleaned comment.

        img_path = os.path.join(self.img_source, f"{name}")
        img = gdal.Open(img_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # NOTE: cleaned English comment.
        img_width = img.RasterXSize
        img_height = img.RasterYSize
        img_data = img.ReadAsArray(0, 0, img_width, img_height)
        del img  # NOTE: cleaned comment.


        # NOTE: cleaned English comment.
        seg_path = os.path.join(self.label_dir, f"{name}")
        seg = gdal.Open(seg_path)
        if seg is None:
            raise FileNotFoundError(f"Label file not found: {seg_path}")

        seg_data = seg.ReadAsArray(0, 0, seg.RasterXSize, seg.RasterYSize)
        del seg
        # NOTE: cleaned English comment.
        if len(seg_data.shape) == 3:
            seg_data = seg_data.squeeze(axis=0)
        if self.DA and self.train:
            img_data_aug, seg_data = self.get_random_data(img_data, seg_data)

            annotation_lines_tgr = self.annotation_lines_tgr[index]
            name_DA = annotation_lines_tgr.split()[0]
            tgr_path = os.path.join(self.img_target, f"{name_DA}")
            tgr = gdal.Open(tgr_path)
            tgr_data = tgr.ReadAsArray()  # NOTE: cleaned comment.
            del tgr  # NOTE: cleaned comment.

            # NOTE: cleaned English comment.
            annotation_lines_sor = self.annotation_lines_sor[index]
            name_DA2 = annotation_lines_sor.split()[0]
            sor_path = os.path.join(self.img_sourcebig, f"{name_DA2}")
            sor = gdal.Open(sor_path)
            sor_data = sor.ReadAsArray()  # NOTE: cleaned comment.
            del sor  # NOTE: cleaned comment.
            # NOTE: cleaned English comment.
            if np.random.rand() > 0.5:
                # NOTE: cleaned English comment.
                DA_train_images_patches = sor_data
                DA_train_labels_patches = 0
            else:
                # NOTE: cleaned English comment.
                DA_train_images_patches = tgr_data
                DA_train_labels_patches =1

            return img_data_aug, seg_data, DA_train_images_patches, DA_train_labels_patches
        else:
            return img_data, seg_data

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, img, label):
        """
        Simple augmentation used during training.
        Applies random flip and fixed-angle rotation (90/180/270).
        Returns image as float32 and label as uint8.
        """
        C, H, W = img.shape  # NOTE: cleaned comment.

        # ------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------#
        img = img.astype(np.float32)
        # NOTE: cleaned English comment.
        label = label.astype(np.uint8)

        # ------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------#
        if self.rand() < 0.5:
            # NOTE: cleaned English comment.
            for c in range(C):
                img[c, :, :] = cv2.flip(img[c, :, :], 1)  # NOTE: cleaned comment.
            # NOTE: cleaned English comment.
            label = cv2.flip(label, 1)

        # ------------------------------#
        # NOTE: cleaned English comment.
        # NOTE: cleaned English comment.
        # ------------------------------#
        rotate_choice = np.random.choice([0, 1, 2, 3])  # NOTE: cleaned comment.
        if rotate_choice != 0:
            # NOTE: cleaned English comment.
            def rotate_channel(channel, choice):
                if choice == 1:  # NOTE: cleaned comment.
                    return cv2.rotate(channel, cv2.ROTATE_90_CLOCKWISE)
                elif choice == 2:  # NOTE: cleaned comment.
                    return cv2.rotate(channel, cv2.ROTATE_180)
                elif choice == 3:  # NOTE: cleaned comment.
                    return cv2.rotate(channel, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    return channel

            # NOTE: cleaned English comment.
            rotated_channels = []
            for c in range(C):
                rotated_channel = rotate_channel(img[c, :, :], rotate_choice)
                rotated_channels.append(rotated_channel)
            img = np.stack(rotated_channels, axis=0)  # NOTE: cleaned comment.

            # NOTE: cleaned English comment.
            label = rotate_channel(label, rotate_choice)

        # ------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------#
        img = img.astype(np.float32)
        label = label.astype(np.uint8)  # NOTE: cleaned comment.
        return img, label




def histogram(array):
    for key, value in sorted(Counter(np.reshape(array, (-1))).items()):
        print("{} : {}".format(key, value))


    def get_random_data_none(self, img_t0, img_t1, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3):

        img_t0 = np.transpose(img_t0,(1, 2, 0))  # NOTE: cleaned comment.
        img_t1 = np.transpose(img_t1,(1, 2, 0))  # NOTE: cleaned comment.
        img_t0 = Image.fromarray(img_t0)  # NOTE: cleaned comment.
        img_t1 = Image.fromarray(img_t1)

        label = Image.fromarray(label.astype(np.uint8))  # NOTE: cleaned comment.

        # ------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------#
        iw, ih = img_t0.size
        h, w = input_shape
        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(1, 2)

        nw = int(scale * w)
        nh = int(nw / new_ar)
        img_t0 = img_t0.resize((nw, nh), Image.BICUBIC)
        img_t1 = img_t1.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            img_t0 = img_t0.transpose(Image.FLIP_LEFT_RIGHT)
            img_t1 = img_t1.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))

        # NOTE: cleaned English comment.
        new_image_t0 = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        new_image_t1 = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        new_image_t0.paste(img_t0, (dx, dy))  # NOTE: cleaned comment.
        new_image_t1.paste(img_t1, (dx, dy))

        image_t0 = new_image_t0
        image_t1 = new_image_t1
        new_label = Image.new('L', (w, h), (0))
        new_label.paste(label, (dx, dy))
        label = new_label

        image_data_t0 = np.array(image_t0, np.uint8)
        image_data_t1 = np.array(image_t1, np.uint8)
        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        blur = self.rand() < 0.25
        if blur:
            image_data_t0 = cv2.GaussianBlur(image_data_t0, (5, 5), 0)
            image_data_t1 = cv2.GaussianBlur(image_data_t1, (5, 5), 0)

        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            # image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))

            image_data_t0 = cv2.warpAffine(
                image_data_t0, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderValue=(0, 0, 0, 0)  # NOTE: cleaned comment.
            )
            image_data_t1 = cv2.warpAffine(
                image_data_t1, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderValue=(0, 0, 0, 0)
            )
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        # ---------------------------------#
        # NOTE: cleaned English comment.
        # NOTE: cleaned English comment.
        # ---------------------------------#


        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data_t0, cv2.COLOR_RGB2HSV))
        dtype = image_data_t0.dtype
            # # ---------------------------------#
            # NOTE: cleaned English comment.
            # # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        img_t0 = np.transpose(image_data_t0, (2, 0, 1))
        img_t1 = np.transpose(image_data_t1, (2, 0, 1))

        return img_t0,img_t1, label


    def get_random_data(self, img_t0, img_t1, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3):

        img_t0 = np.transpose(img_t0,(1, 2, 0))  # NOTE: cleaned comment.
        img_t1 = np.transpose(img_t1,(1, 2, 0))  # NOTE: cleaned comment.


        # ------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------#
        iw, ih,_ = img_t0.shape
        h, w = input_shape
        # ------------------------------------------#
        # NOTE: cleaned English comment.
        # ------------------------------------------#
        new_ar = iw / ih
        scale = self.rand(1, 2)

        nw = int(scale * w)
        nh = int(nw / new_ar)
        img_t0 = cv2.resize(img_t0, (nw, nh), interpolation=cv2.INTER_CUBIC)
        img_t1 = cv2.resize(img_t1, (nw, nh), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if nw > w:
            x1 = np.random.randint(0, nw - w)  # NOTE: cleaned comment.
        else:
            x1 = 0  # NOTE: cleaned comment.

        if nh > h:
            y1 = np.random.randint(0, nh - h)  # NOTE: cleaned comment.
        else:
            y1 = 0  # NOTE: cleaned comment.

        # NOTE: cleaned English comment.
        img_t0 = img_t0[y1:y1 + h, x1:x1 + w, :]  # NOTE: cleaned comment.
        img_t1 = img_t1[y1:y1 + h, x1:x1 + w, :]
        label = label[y1:y1 + h, x1:x1 + w]

        blur = self.rand() < 0.25
        if blur:
            img_t0 = cv2.GaussianBlur(img_t0, (5, 5), 0)
            img_t1 = cv2.GaussianBlur(img_t1, (5, 5), 0)
        flip = self.rand() < .5
        if flip:
            img_t0 = cv2.flip(img_t0, 1)  # NOTE: cleaned comment.
            img_t1 = cv2.flip(img_t1, 1)
            label = cv2.flip(label, 1)

        # --------------------------
        # NOTE: cleaned English comment.
        # --------------------------
        flip2 = self.rand() < .5
        if flip2:
            img_t0 = cv2.flip(img_t0, 0)  # NOTE: cleaned comment.
            img_t1 = cv2.flip(img_t1, 0)
            label = cv2.flip(label, 0)

        # --------------------------
        # NOTE: cleaned English comment.
        # --------------------------
        rot = self.rand() < .5
        if rot:
            k = random.choice([1, 2, 3])  # NOTE: cleaned comment.
            img_t0 = np.rot90(img_t0, k)
            img_t1 = np.rot90(img_t1, k)
            label = np.rot90(label, k)
            # NOTE: cleaned English comment.
            img_t0 = cv2.resize(img_t0, (h, w))
            img_t1 = cv2.resize(img_t1, (h, w))
            label = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST)  # NOTE: cleaned comment.

        # --------------------------
        # NOTE: cleaned English comment.
        # --------------------------
        img_t0 = np.transpose(img_t0, (2, 0, 1))  # NOTE: cleaned comment.
        img_t1 = np.transpose(img_t1, (2, 0, 1))


        return img_t0, img_t1, label
