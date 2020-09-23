"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os


import os.path
from data.coco_dataset import CocoDataset
from data.image_folder import make_dataset
import torch
import random

import numpy as np 
import cv2


class LandscapesDataset(CocoDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = CocoDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=17)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(aspect_ratio=1.0)
        opt, _ = parser.parse_known_args()
        # if hasattr(opt, 'num_upsampling_layers'):
            # parser.set_defaults(num_upsampling_layers='more')

        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.class_of_interest = [96, 105, 118, 123, 125, 126, 134, 135, 147, 149, 153, 154, 156, 158, 168, 177, 181, 255] # will define it in child
        self.class_dict = dict(zip(self.class_of_interest, list(range(len(self.class_of_interest)))))

        if opt.load_masks:
            self.mask_paths =  make_dataset("/home/ens/data/irregular_mask")
            self.mask_len = len(self.mask_paths)
            self.paint_mask = False
        else:
            self.paint_mask = opt.phase == "test"

        self.label_nc = opt.label_nc


 
    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase

        data_dir = os.path.join(root, phase)
        paths = make_dataset(data_dir, recursive=True)

        label_paths = []
        image_paths = []

        for p in paths:
            if p.endswith("jpg"):
                image_paths.append(p)
            elif p.endswith("png"):
                label_paths.append(p)
            else:
                raise ValueError("Dataset doesn't contain only images jpg/png")

        assert opt.no_instance 
        instance_paths = []

        return label_paths, image_paths, instance_paths

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)
        # Give subclasses a chance to modify the final output

        # new_label = torch.tensor(14).expand(label_tensor.shape)
        if self.paint_mask:
            mask_tensor = (label_tensor == 0).float()
            label_tensor[label_tensor == 0] = 254

        for old_i, new_i in self.class_dict.items():
            label_tensor[label_tensor == old_i] = new_i
        
        label_tensor[label_tensor > self.label_nc] = self.label_nc 
 
        if self.opt.load_masks:
            # mask_path = random.choice(self.mask_paths)
            # mask = Image.open(mask_path)
            # mask_tensor = transform_label(mask) 
            if random.random() > .3:
                mask_tensor = torch.tensor(self.random_ff_mask(label_tensor)).unsqueeze(0)
                mask_tensor =  mask_tensor * self.get_mask(label_tensor).float()
            else:
                mask_tensor =  1 -(label_tensor == random.choice(label_tensor.unique())).float()

            if self.opt.roll_masks:
                roll_n = random.randint(-10,10)
                roll_dim = random.randint(1,2)
                mask_tensor = torch.roll(mask_tensor, roll_n, roll_dim)
                label_tensor = torch.roll(label_tensor, roll_n, roll_dim)
                        
 
        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'mask': mask_tensor,
                      }

        return input_dict



    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return name1[:3] == name2[:3]


    # From generative inpainting code: https://github.com/JiahuiYu/generative_inpainting
    @staticmethod
    def random_ff_mask(x):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """

        _, h, w = x.shape
        mask = np.zeros((h,w))
        num_v = 12+np.random.randint(5)#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(4.0)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(40)
                brush_w = 10+np.random.randint(10)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return 1 - mask.astype(np.float32)

    @staticmethod
    def get_mask(x):
        _, h, w = x.shape
        n = 1
        r = 1 - ((torch.rand(n,h+1).sort(1).indices <2).long().cumsum(1)-1).abs()[:,:h]
        c = 1 - ((torch.rand(n,w+1).sort(1).indices <2).long().cumsum(1)-1).abs()[:,:w]
        mask = (r.view(1,h,1) * c.view(1,1,w)).expand_as(x)
        return 1 - mask
