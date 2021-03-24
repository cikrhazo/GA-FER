import torch
from torch import nn
from network.blocks import Spatial_Basic_Block, Spatial_Bottleneck_Block
import logging
from network.Cattention import Comp_Att
from datasets.graphs import Graph


class ResGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, block, A, initial=False, stride=1, kernel_size=[9, 2]):
        super(ResGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if initial:
            module_res, block_res = False, False
        elif block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        if block == "Basic":
            spatial_block = Spatial_Basic_Block
            # temporal_block = Temporal_Basic_Block
        else:
            spatial_block = Spatial_Bottleneck_Block
            # temporal_block = Temporal_Bottleneck_Block
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, block_res)
        # self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res)
        self.edge = nn.Parameter(torch.ones_like(A), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        return self.relu(self.scn(x, A * self.edge) + self.residual(x))


class AttGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, block, A, attention=Comp_Att, stride=1, kernel_size=[9,2]):
        super(AttGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )

        if block == "Basic":
            spatial_block = Spatial_Basic_Block
        else:
            spatial_block = Spatial_Bottleneck_Block
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, block_res)
        self.att = attention(out_channels, parts=Graph().parts)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, A):
        return self.att(self.scn(x, A*self.edge))
