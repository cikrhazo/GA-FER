from network.net_gcn import ResGCN
from network.net_cnn_single import VisModel
import torch
from torch import nn
import torch.nn.functional as F


class TowStream(nn.Module):

    def __init__(self, module, structure, block, A, data_shape, num_class, mem_size):
        super(TowStream, self).__init__()

        self.gcn = ResGCN(
            module=module,
            structure=structure,
            block=block,
            A=A,
            data_shape=data_shape,
            num_class=num_class,
            mem_size=mem_size
        )

        self.cnn = VisModel(num_class=num_class)

        self.aff = AttFeaFusion()

        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_class)
        )

    def forward(self, geo_tensor, vis_tensor, img_tensor):

        _, feat_geo = self.gcn(geo_tensor, vis_tensor)
        _, feat_vis = self.cnn(img_tensor)

        feature = self.aff(feat_geo, feat_vis)

        out = self.classifier(feature)
        return out, feature


class AttFeaFusion(nn.Module):
    def __init__(self, in_dim=256, out_dim=512):
        super(AttFeaFusion, self).__init__()

        self.geo_enc = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.vis_enc = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.shortcut = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

        self.aff = nn.Sequential(
            nn.Linear(out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, geo, vis):
        geo_ = self.geo_enc(geo.unsqueeze(-1).unsqueeze(-1))
        vis_ = self.vis_enc(vis.unsqueeze(-1).unsqueeze(-1))

        fus_ = torch.cat((geo_, vis_), dim=1).squeeze(-1).squeeze(-1)
        fus_ = self.aff(fus_)

        ffatt = F.softmax(fus_, dim=1)
        ffatt = torch.repeat_interleave(ffatt, repeats=256, dim=1)

        fusion = torch.cat((geo, vis), dim=1) * ffatt

        res = self.shortcut(torch.cat((geo, vis), dim=1))
        fusion = fusion + res

        return F.relu(fusion)
