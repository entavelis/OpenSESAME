"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.util as util

class SesameMultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='sesame_n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, input_nc = None):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, input_nc)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, input_nc = None):
        subarch = opt.netD_subarch
        if subarch == 'sesame_n_layer':
            netD = SesameNLayerDiscriminator(opt, input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the SESAME discriminator with the specified arguments.
class SesameNLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, input_nc=None):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        if input_nc is None:
            input_nc = self.compute_D_input_nc(opt)

        branch = []
        sizes = (input_nc - 3, 3) 
        original_nf = nf
        for input_nc in sizes: 
            nf = original_nf
            norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
            sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.2, False)]]

            for n in range(1, opt.n_layers_D):
                nf_prev = nf
                nf = min(nf * 2, 512)
                stride = 1 if n == opt.n_layers_D - 1 else 2
                sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False)
                              ]]

            branch.append(sequence)
            
        sem_sequence = nn.ModuleList()
        for n in range(len(branch[0])):
            sem_sequence.append(nn.Sequential(*branch[0][n]))
        self.sem_sequence = nn.Sequential(*sem_sequence)

        sequence = branch[1]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        self.img_sequence = nn.ModuleList()
        for n in range(len(sequence)):
            self.img_sequence.append(nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        label_nc = opt.label_nc
        input_nc = label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if not opt.no_inpaint:
            input_nc += 1
            
        return input_nc

    def forward(self, input):
        img, sem = input[:,-3:], input[:,:-3]
        sem_results = self.sem_sequence(sem)
        results = [img]
        for submodel in self.img_sequence[:-1]:
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        intermediate_output = self.my_dot(intermediate_output, sem_results)
        results.append(self.img_sequence[-1](intermediate_output))

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

    def my_dot(self, x, y):
        return x + x * y.sum(1).unsqueeze(1)