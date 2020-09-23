""".unsqueeze(0)
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torchvision.transforms as t 
import models.networks as networks
import util.util as util
from random import randint, random

import numpy as np

class Pix2PixModel(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_style_loss:
                self.criterionStyle = networks.StyleLoss(self.opt.gpu_ids)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image, masked_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, masked_image)
            return g_loss, generated, masked_image, input_semantics
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, masked_image)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image = self.generate_fake(input_semantics, real_image, masked_image)
            return fake_image, masked_image, input_semantics
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())

        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    # preprocess the input, such as moving the tensors to GPUs and`
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()

        if not self.opt.bbox:
            if self.use_gpu():
                label = data['label'].cuda()
                inst = data['instance'].cuda()
                image =  data['image'].cuda()
                masked = data['image'].cuda()
            else:
                label = data['label']
                inst = data['instance']
                image =  data['image']
                masked = data['image']
       
        else:
            label = data['label'].cuda()
            inst = data['inst'].cuda()
            image = data['image'].cuda()
            mask_in = data['mask_in'].cuda()
            mask_out = data['mask_out'].cuda()
            masked = data['image'].cuda()

        # Get Semantics
        input_semantics = self.get_semantics(label)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = inst
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        if not self.opt.no_inpaint:
            if "mask" in data.keys():
                mask = data["mask"]
                if self.use_gpu():
                    mask = mask.cuda()

            elif self.opt.bbox:
                mask = 1 - mask_in
                   
            else:
                # Cellural Mask used for AIM2020 Challenge on Image Extreme Inpainting
                mask = self.get_mask(image.size())


            assert input_semantics.sum(1).max() == 1
            assert input_semantics.sum(1).min() == 1

            masked =  image * mask
            if self.opt.segmentation_mask:
                input_semantics *= (1-mask)


            input_semantics = torch.cat([input_semantics,1-mask[:,0:1]],dim=1)


        return input_semantics, image, masked

    def compute_generator_loss(self, input_semantics, real_image, masked_image):
        G_losses = {}

        fake_image = self.generate_fake(
            input_semantics, real_image, masked_image)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        if self.opt.use_style_loss:
            G_losses['Style'] = self.criterionStyle(fake_image, real_image) \
                * self.opt.lambda_style

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, masked_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(input_semantics, real_image, masked_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        mask = input_semantics[:,[-1]]


        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def generate_fake(self, input_semantics, real_image, masked_image = None):
        if not self.opt.no_inpaint:
            mask = input_semantics[:,[-1]]
            fake_image = self.netG(masked_image, input_semantics)

            if not self.opt.no_mix_real_fake:
                fake_image = (1 - mask) * real_image + mask * fake_image 
        else:
            fake_image = self.netG(input_semantics, masked_image)


        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        # if self.opt.netG == 'unet':
        #     input_semantics = input_semanno_tics[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]
        #     fake_image = fake_image[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]
        #     real_image = real_image[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]




        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def get_mask(self, size, times = 7):
        # mask = torch.ones_like(data['image'])
        scale = 4
        b,_,x,y = size
        mask = torch.rand(b,1,x//scale,y//scale).cuda()
        pool = torch.nn.AvgPool2d(3, stride=1,padding=1)
        mask = (mask > 0.5).float()

        for i in range(times):
            mask = pool(mask)
            mask = (mask > 0.5).float()
        
        if scale > 1:
            mask = torch.nn.functional.interpolate(mask, size=(x,y))

        return 1 - mask

    def get_semantics(self, label_map):
        # create one-hot label map
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        return input_semantics
            

