import torch
from torch import nn


class channel_ap(nn.Module):
    def __init__(self):
        super(channel_ap, self).__init__()

    def forward(self, x):
        y = torch.mean(x, dim=1, keepdim=True)
        return y


class channel_mp(nn.Module):
    def __init__(self):
        super(channel_mp, self).__init__()

    def forward(self, x):
        y = torch.max(x, dim=1, keepdim=True)
        return y[0]


class Comp_Att(nn.Module):
    def __init__(self, channel, parts, **kwargs):
        super(Comp_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.conv1 = nn.Conv2d(channel, inter_channel, kernel_size=1)
        self.Ch_AP = channel_ap()
        self.Ch_MP = channel_mp()
        self.covn2 = nn.Conv2d(2, 1, kernel_size=1, padding=0, stride=1)

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_parts = [torch.mean(x[:, :, :, self.joints == i], 3, True) for i in range(len(self.parts))]  # N C T U
        x_parts = torch.cat(x_parts, dim=-1)  # N C T U
        x_parts = self.conv1(x_parts)
        x_cmp = self.Ch_MP(x_parts)
        x_cap = self.Ch_AP(x_parts)
        pp = torch.cat((x_cap, x_cmp), dim=1)
        x_att = self.softmax(self.covn2(pp).repeat(1, C, 1, 1).view(N, C, len(self.parts)))

        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:, :, :, i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)

        return self.relu(self.bn(x * x_att) + res)


def get_corr_joints(parts):
    num_joints = max([max(part) for part in parts]) + 1
    res = []
    for i in range(num_joints):
        for j in range(len(parts)):
            if i in parts[j]:
                res.append(j)
                break
    return torch.Tensor(res).long()
