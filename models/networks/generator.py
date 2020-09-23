""" Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

class SesameGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=2, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=4, help='number of residual blocks in the global generator network')
        parser.add_argument('--spade_n_blocks', type=int, default=5, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = 3
        label_nc = opt.label_nc

        input_nc = label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        if opt.mix_input_gen:
            input_nc += 4

        norm_layer = get_nonspade_norm_layer(opt, 'instance')
        activation = nn.ReLU(False)


        # initial block 
        self.init_block = nn.Sequential(*[nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation])
        
        # Downsampling blocks
        self.downlayers = nn.ModuleList()
        mult = 1
        for i in range(opt.resnet_n_downsample):
            self.downlayers.append(nn.Sequential(*[norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]))
            mult *= 2

        # Semantic core blocks
        self.resnet_core = nn.ModuleList()
        if opt.wide: 
            self.resnet_core += [ResnetBlock(opt.ngf * mult,
                                dim2=opt.ngf * mult * 2,
                                norm_layer=norm_layer,
                                activation=activation,
                                kernel_size=opt.resnet_kernel_size)]
            mult *= 2
        else:
            self.resnet_core += [ResnetBlock(opt.ngf * mult,
                                norm_layer=norm_layer,
                                activation=activation,
                                kernel_size=opt.resnet_kernel_size)]


        for i in range(opt.resnet_n_blocks - 1):
            self.resnet_core += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size,
                                  dilation=2)]


        self.spade_core = nn.ModuleList()
        for i in range(opt.spade_n_blocks - 1):
            self.spade_core += [SPADEResnetBlock(opt.ngf * mult, opt.ngf * mult, opt, dilation=2)]

        if opt.wide:
            self.spade_core += [SPADEResnetBlock(opt.ngf * mult * (2 if not self.opt.no_skip_connections else 1), opt.ngf * mult//2, opt)]
            mult//=2
        else:
            self.spade_core += [SPADEResnetBlock(opt.ngf * mult * (2 if not self.opt.no_skip_connections else 1), opt.ngf * mult, opt)]

        # Upsampling blocks
        self.uplayers = nn.ModuleList()
        for i in range(opt.resnet_n_downsample):
            self.uplayers.append(SPADEResnetBlock(mult * opt.ngf * (3 if not self.opt.no_skip_connections else 2)//2, opt.ngf * mult//2, opt))
            mult //= 2

        final_nc = opt.ngf


        self.conv_img = nn.Conv2d((input_nc + final_nc) if not self.opt.no_skip_connections else final_nc , output_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, input, extra=None):
        if self.opt.mix_input_gen:
            input = torch.cat([input,extra], dim = 1)
        
        dec_i = self.opt.resnet_n_downsample 
        skip_connections = []
        # InitBlock
        x = self.init_block(input)
        skip_connections.append(x)
        # /InitBlock

        # Downsize
        for downlayer in self.downlayers:
            x = downlayer(x)
            skip_connections.append(x)
        # /Downsize

        # SemanticCore 
        for res_layer in self.resnet_core:
            x = res_layer(x)


        for spade_layer in self.spade_core[:-1]:
            x = spade_layer(x, extra)
            
        if not self.opt.no_skip_connections:
            x = torch.cat([x, skip_connections[dec_i]],dim=1)
            dec_i -= 1
    
        x = self.spade_core[-1](x, extra)
        # /SemanticCore 

        # Upsize
        for uplayer in self.uplayers:
            x = self.up(x)
            if not self.opt.no_skip_connections:
                x = torch.cat([x, skip_connections[dec_i]],dim=1)
                dec_i -= 1
            x = uplayer(x, extra)
        # /Upsize


        # OutBlock
        if not self.opt.no_skip_connections:
            x = torch.cat([x, input],dim=1)
        x = self.conv_img(F.leaky_relu(x, 2e-1))

        x = F.tanh(x)
        # /OutBlock

        return x