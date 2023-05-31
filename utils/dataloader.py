import cv2
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import torch.utils.data
import xml.etree.ElementTree as ET

from utils.utils import cvtColor, preprocess_input


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape = [600, 600], train = True):
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, y    = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data    = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box         = box_data[:, :4]
        label       = box_data[:, -1]
        return image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = torch.from_numpy(np.array(images))
    return images, bboxes, labels

class vocdataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train=True, data_dir=None, transform=None, target_transform=None, keep_difficult=False):
        """VOC格式数据集
        Args:
            data_dir: VOC格式数据集根目录，该目录下包含：
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
            split： train、test 或者 eval， 对应于 ImageSets/Main/train.txt,eval.txt
        """
        # 类别
        self.class_names = cfg.DATA.DATASET.CLASS_NAME
        self.data_dir = cfg.DATA.DATASET.DATA_DIR
        self.is_train = is_train
        if data_dir:
            self.data_dir = data_dir
        self.split = cfg.DATA.DATASET.TRAIN_SPLIT       # train     对应于ImageSets/Main/train.txt
        if not self.is_train:
            self.split = cfg.DATA.DATASET.TEST_SPLIT    # test      对应于ImageSets/Main/test.txt
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "{}.txt".format(self.split))
        # 从train.txt 文件中读取图片 id 返回ids列表
        self.ids = self._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_name = self.ids[index]
        # 解析Annotations/id.xml 读取id图片对应的 boxes, labels, is_difficult 均为列表
        boxes, labels, is_difficult = self._get_annotation(image_name)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        # 读取 JPEGImages/id.jpg 返回Image.Image
        image = self._read_image(image_name)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels, image_name

    # 返回 id, boxes， labels， is_difficult
    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    # 解析xml，返回 boxes， labels， is_difficult   numpy.array格式
    def _get_annotation(self, image_name):
        annotation_file = os.path.join(self.data_dir, "Annotations", "{}.xml".format(image_name))
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects: # .encode('utf-8').decode('UTF-8-sig') 解决Windows下中文编码问题
            class_name = obj.find('name').text.encode('utf-8').decode('UTF-8-sig').lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text.encode('utf-8').decode('UTF-8-sig')) - 1
            y1 = float(bbox.find('ymin').text.encode('utf-8').decode('UTF-8-sig')) - 1
            x2 = float(bbox.find('xmax').text.encode('utf-8').decode('UTF-8-sig')) - 1
            y2 = float(bbox.find('ymax').text.encode('utf-8').decode('UTF-8-sig')) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    # 获取图片尺寸信息，返回字典 {'height': , 'width': }
    def get_img_size(self, img_name):
        annotation_file = os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_name))
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    # 读取图片数据，返回Image.Image
    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "{}.jpg".format(image_id))
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def get_one_image(self,image_name = None):
        import random

        if not image_name:
            image_name = random.choice(self.ids)
        # 解析Annotations/id.xml 读取id图片对应的 boxes, labels, is_difficult 均为列表
        boxes, labels, is_difficult = self._get_annotation(image_name)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        # 读取 JPEGImages/id.jpg 返回Image.Image
        image = self._read_image(image_name)
        image_after_transfrom = None
        if self.transform:
            image_after_transfrom, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, image_after_transfrom, boxes, labels, image_name
    
class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[3])

        if self.is_train:
            boxes = default_collate(transposed_batch[1])
            labels = default_collate(transposed_batch[2])
        else:
            boxes = None
            labels = None
        return images, boxes, labels, img_ids