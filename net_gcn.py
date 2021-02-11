import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from network.modules import ResGCN_Module, AttGCN_Module
from mmcv.cnn import constant_init, kaiming_init
from torch.cuda.amp import autocast

###########################################################################################################

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

###########################################################################################################


class ResGCN_Input_Branch(nn.Module):
    def __init__(self, structure, block, num_channel, A):
        super(ResGCN_Input_Branch, self).__init__()

        self.register_buffer('A', A)

        module_list = [ResGCN_Module(num_channel, 64, 'Basic', A, initial=True)]
        module_list += [ResGCN_Module(64, 64, 'Basic', A, initial=True) for _ in range(structure[0] - 1)]
        module_list += [ResGCN_Module(64, 64, block, A) for _ in range(structure[1] - 1)]
        module_list += [ResGCN_Module(64, 32, block, A)]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        N, C, T, P = x.size()  # batch * coordinate * frame * point
        x = self.bn(x)  # batch * coordinate * frame * point
        for layer in self.layers:
            x = layer(x, self.A)

        return x


class ResGCN(nn.Module):
    def __init__(self, module, structure, block, data_shape, num_class, A, mem_size=64):
        super(ResGCN, self).__init__()

        num_input, num_channel, _, _ = data_shape  # num_input is how many branches of input
        self.register_buffer('A', A)

        # input branches
        self.InputBranchGeo = nn.ModuleList([
            ResGCN_Input_Branch(structure, block, num_channel, A)
            for _ in range(num_input)
        ])
        point = int(A.size(-1))
        self.InputBranchVis = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(3, mem_size, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(mem_size),
                nn.ReLU(inplace=True),
                # pool:
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(mem_size, mem_size, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(mem_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(mem_size, mem_size, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(mem_size),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(mem_size, mem_size, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(mem_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(mem_size, mem_size, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(mem_size),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(mem_size, 32, kernel_size=6, padding=0, stride=1),
            )
            for _ in range(point)
        )
        # self.abstract = nn.Conv2d(mem_size, 32, kernel_size=6, padding=0, stride=1)
        self.VisionRes = nn.Sequential(
            nn.Linear(32 * point, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256 * point),
        )

        # self.GeoRes = nn.Sequential(
        #     nn.Linear(32 * 3 * point, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256 * point),
        # )

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(1, rgb_mean, rgb_std)

        # main stream
        module_list = [module(32 * num_input + 32, 128, block, A, stride=2)]
        module_list += [module(128, 128, block, A) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, block, A, stride=2)]
        module_list += [module(256, 256, block, A) for _ in range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)

        # output
        self.global_pooling = nn.Linear(256 * point, 256)
        self.fcn = nn.Linear(256, num_class)

        # init parameters
        self._init_weights()
        self.zero_init_lastBN()

    @autocast()
    def forward(self, g, v):

        N, I, C, T, V = g.size()  # batch * branch * coordinate * frame * point

        # input branches
        g_cat = []
        # Todo multiprocess
        for i, branch in enumerate(self.InputBranchGeo):
            g_cat.append(branch(g[:, i, :, :, :]))
        g = torch.cat(g_cat, dim=1)

        Bz, P, Ch, H, W = v.size()
        v = self.sub_mean(v.view(Bz * P, Ch, H, W))
        v = v.view(Bz, P, Ch, H, W)
        v_cat = []
        # Todo multiprocess
        # Using 3D Conv can speed up the operation but the 3D bacth norm is not correct, which may lead to a performance degardation.
        for i, branch in enumerate(self.InputBranchVis):
            v_cat.append(branch(v[:, i, :, :, :]))
        v = torch.stack(v_cat, dim=4)
        v = v.view(Bz, -1, 1, 1, P).squeeze(-2).contiguous()  # batch * channel * frame * point

        # fusion
        x = torch.cat((g, v), dim=1)

        # main stream
        for layer in self.main_stream:
            x = layer(x, self.A)

        # extract feature
        _, C, T, V = x.size()
        feature = x.view(N, C, T, V)
        v_res = self.VisionRes(v.view(N, -1))

        # output
        feature = self.global_pooling(feature.view(N, -1) + v_res)
        feature = F.relu(feature.view(N, -1))
        x = self.fcn(feature)

        return x, feature

    # warm-up the network
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # m.bias = None
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def zero_init_lastBN(self):
        for m in self.modules():
            if isinstance(m, ResGCN_Module):
                if hasattr(m.scn, 'bn_up'):
                    nn.init.constant_(m.scn.bn_up.weight, 0)


if __name__ == "__main__":
    from torch.autograd import Variable
    from datasets.graphs import Graph
    from network.modules import ResGCN_Module

    A = torch.from_numpy(Graph().A.astype(np.float32))
    A = Variable(A.cuda(), requires_grad=False)

    gcn = ResGCN(
        module=AttGCN_Module,
        structure=[1, 2, 3, 3],
        block='Bottleneck',
        A=A,
        data_shape=(3, 2, 1, 68),
        num_class=7,
        mem_size=64
    )
    gcn.cuda()

    g = torch.randn(size=(3, 3, 2, 1, 68))  # batch * branch * coordinate * frame * point
    g = Variable(g.cuda(), requires_grad=False)

    v = torch.randn(size=(3, 68, 3, 49, 49))
    v = Variable(v.cuda(), requires_grad=False)

    out_tensor = gcn(g, v)[0]
    print(out_tensor.requires_grad)
