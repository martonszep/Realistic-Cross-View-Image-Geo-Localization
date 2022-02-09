import torch.nn as nn
import torch.nn.functional as F
from networks import backbones, spatial_transf
import torch

class SA(nn.Module):
    def __init__(self, in_dim, num=8):
        super().__init__()
        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, num)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, num)

    def init_weights_(self, din, dout, dnum):
        weight = torch.empty(din, dout, dnum)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        bias = torch.empty(1, dout, dnum)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        mask, _ = x.max(1)
        
        mask = torch.einsum('bi, ijd -> bjd', mask, self.w1) + self.b1 
        mask = torch.einsum('bjd, jid -> bid', mask, self.w2) + self.b2
        return mask


# Siamese network w/o shared weights
class SAFA(nn.Module):
    #Modified version of "Spatial-Aware Feature Aggregation for Cross-View Image Based Geo-Localization" paper
    def __init__(self, sa_num=8, H1=112, W1=616, H2=112, W2=616, use_spatialtr=False, use_tps=True):
        super().__init__()

        self.spatial_tr = spatial_transf.ComposedSpatialTransf(in_channels=3, spatial_dims=(H2, W2), use_tps=use_tps) if use_spatialtr else None
        self.transformed_satellite = None

        self.extract1 = backbones.ResNet34()
        self.extract2 = backbones.ResNet34()

        in_dim1 = (H1 // 8) * (W1 // 8)
        in_dim2 = (H2 // 8) * (W2 // 8)
        self.sa1 = SA(in_dim1, sa_num)
        self.sa2 = SA(in_dim2, sa_num)

    def forward(self, street, satellite):

        if self.spatial_tr is not None:
            self.transformed_satellite = self.spatial_tr(satellite)
            satellite = self.transformed_satellite

        # this is necessary because of the different batch sizes of the two image sets in the VIGOR validation code
        if street is None:
            street_extracted = None
        else:
            # Local feature extraction
            street_extracted = self.extract1(street)

            B_street, C, _, _ = street_extracted.shape
            street_extracted = street_extracted.view(B_street, C, -1)

            # Spatial aware attention
            w1 = self.sa1(street_extracted)

            # Global feature aggregation
            street_extracted = torch.matmul(street_extracted, w1).view(B_street, -1)

            # Feature reduction
            street_extracted = F.normalize(street_extracted, p=2, dim=1)

        # this is necessary because of the different batch sizes of the two image sets in the VIGOR validation code
        if satellite is None:
            sat_extracted = None
        else:
            # Local feature extraction
            sat_extracted = self.extract2(satellite)

            B_sat, C, _, _ = sat_extracted.shape
            sat_extracted = sat_extracted.view(B_sat, C, -1)

            # Spatial aware attention
            w2 = self.sa2(sat_extracted)

            # Global feature aggregation
            sat_extracted = torch.matmul(sat_extracted, w2).view(B_sat, -1)

            # Feature reduction
            sat_extracted = F.normalize(sat_extracted, p=2, dim=1)

        return sat_extracted, street_extracted