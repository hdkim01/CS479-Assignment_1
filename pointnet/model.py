import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.max(x, 2)[0]
        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x
    


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.

        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))
       

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.

        if self.input_transform:
            pointcloud = pointcloud.transpose(2, 1)
            transform3_3 = self.stn3(pointcloud)
            pointcloud = pointcloud.transpose(2, 1)
            pointcloud = torch.bmm(pointcloud, transform3_3)
        
        pointcloud = pointcloud.transpose(2, 1)
        pointcloud = self.mlp1(pointcloud)

        if self.feature_transform:
            transform64_64 = self.stn64(pointcloud)
            pointcloud = pointcloud.transpose(2, 1)
            trans_feat2 = torch.bmm(pointcloud, transform64_64)
            pointcloud = trans_feat2.transpose(2, 1)
        else:
            transform64_64, trans_feat2 = None, None

        global_latent = self.mlp2(pointcloud)

        global_feat = torch.max(global_latent, 2)[0]

        return global_feat, transform64_64, trans_feat2, global_latent




class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.mlp = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
                                 nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256, self.num_classes), nn.BatchNorm1d(self.num_classes))
                                 

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.

        global_feat, transform64_64 , _ , _ = self.pointnet_feat(pointcloud)
        
        global_feat = global_feat.unsqueeze(-1)
        
        global_feat = self.mlp(global_feat)
        
        global_feat = global_feat.squeeze(-1)
        
        output = self.fc(global_feat)
        
        
        return output, transform64_64


"""point = torch.randn(8, 60, 3)

net = PointNetCls(num_classes=40, input_transform=True, feature_transform=True)
p = net(point)
print(p.size())"""


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.pointnet_feat = PointNetFeat(input_transform=True, feature_transform=True)

        self.mlp = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
                                  nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, m, 1), nn.BatchNorm1d(m))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        _, transform64_64, trans_feat2, global_latent = self.pointnet_feat(pointcloud)
        trans_feat2 = trans_feat2.transpose(2, 1)
        point_wise_feat = torch.cat((trans_feat2, global_latent), dim=1)
        
        output = self.mlp(point_wise_feat)

        return output, transform64_64





class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat(input_transform=False, feature_transform=False)

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.decoder = nn.Sequential(nn.Linear(1024, num_points//4), nn.BatchNorm1d(num_points//4), nn.ReLU(),
                                     nn.Linear(num_points//4, num_points//2), nn.BatchNorm1d(num_points//2), nn.ReLU(),
                                     nn.Linear(num_points//2, num_points), nn.Dropout(p=0.2), nn.BatchNorm1d(num_points), nn.ReLU(),
                                     nn.Linear(num_points, num_points*3))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        B = pointcloud.shape[0]
        global_feature, _, _, _ = self.pointnet_feat(pointcloud)
        pointcloud = self.decoder(global_feature)
        pointcloud = pointcloud.reshape(B, -1, 3)

        return pointcloud

        


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
