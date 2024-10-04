
from __future__ import absolute_import, division, print_function
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import roma

from .netOps.geom_quaternion_toolkit import EulerQuat, GramSchmidtQuat, SVDQuat


class CameraNetQuat(nn.Module):
    '''
    input feature latent vector feat_bxz
    outputs camera_bx7 [scale, trans_x, trans_y, quat]
    '''
    def __init__(self, nz_feat=100, scale_bias=0.75):
        super().__init__()
        self.quat_predictor = QuatNet(nz_feat)
        self.scale_predictor = ScaleNet(nz_feat, bias=scale_bias)
        self.trans_predictor = TransNet(nz_feat)

    def forward(self, feat):
        # shape_pred = self.shape_predictor(feat)
        scale_pred = self.scale_predictor(feat)
        quat_pred = self.quat_predictor(feat)
        trans_pred = self.trans_predictor(feat)

        assert (scale_pred.shape == (feat.shape[0], 1))
        assert (trans_pred.shape == (feat.shape[0], 2))
        assert (quat_pred.shape == (feat.shape[0], 4))
        return torch.cat((scale_pred, trans_pred, quat_pred), dim=1)


class CameraNetEuler(nn.Module):
    '''
    input feature latent vector feat_bxz
    outputs camera_bx7 [scale, trans_x, trans_y, quat]
    '''
    def __init__(self, nz_feat=100, scale_bias=0.75):
        super().__init__()
        self.scale_predictor = ScaleNet(nz_feat, bias=scale_bias)
        self.trans_predictor = TransNet(nz_feat)
        self.rot_predictor = RotationNet(nz_feat, nz_rot=3)
        self.quat_tansformer = EulerQuat()
        self.base_quat = self.quat_tansformer.get_base_quaternions(initial_quat_bias_deg=45.,
                                                                   elevation_bias=0,
                                                                   azimuth_bias=0)

    def forward(self, feat):
        scale_pred = self.scale_predictor(feat)
        trans_pred = self.trans_predictor(feat)
        rot_pred = self.rot_predictor(feat)
        quat_pred = self.quat_tansformer.tranform(rot_pred, self.base_quat)

        assert (scale_pred.shape == (feat.shape[0], 1))
        assert (trans_pred.shape == (feat.shape[0], 2))
        assert (quat_pred.shape == (feat.shape[0], 4))

        return torch.cat((scale_pred, trans_pred, quat_pred), dim=1)


class CameraNetGramSchmidt(nn.Module):
    '''
    input feature latent vector feat_bxz
    outputs camera_bx7 [scale, trans_x, trans_y, quat]
    '''
    def __init__(self, nz_feat=100, scale_bias=0.75):
        super().__init__()
        self.scale_predictor = ScaleNet(nz_feat, bias=scale_bias)
        self.trans_predictor = TransNet(nz_feat)
        self.rot_predictor = RotationNet(nz_feat, nz_rot=6)
        self.quat_tansformer = GramSchmidtQuat()

    def forward(self, feat):
        scale_pred = self.scale_predictor(feat)
        trans_pred = self.trans_predictor(feat)
        rot_pred = self.rot_predictor(feat)
        quat_pred = self.quat_tansformer.tranform(rot_pred)

        assert (scale_pred.shape == (feat.shape[0], 1))
        assert (trans_pred.shape == (feat.shape[0], 2))
        assert (quat_pred.shape == (feat.shape[0], 4))

        return torch.cat((scale_pred, trans_pred, quat_pred), dim=1)


class CameraNetSVD(nn.Module):
    '''
    input feature latent vector feat_bxz
    outputs camera_bx7 [scale, trans_x, trans_y, quat]
    '''
    def __init__(self, nz_feat=100, scale_bias=0.75):
        super().__init__()
        self.scale_predictor = ScaleNet(nz_feat, bias=scale_bias)
        self.trans_predictor = TransNet(nz_feat)
        self.rot_predictor = RotationNet(nz_feat, nz_rot=9)
        self.quat_tansformer = SVDQuat()

    def forward(self, feat):
        scale_pred = self.scale_predictor(feat)
        trans_pred = self.trans_predictor(feat)
        rot_pred = self.rot_predictor(feat)
        quat_pred = self.quat_tansformer.tranform(rot_pred)

        assert (scale_pred.shape == (feat.shape[0], 1))
        assert (trans_pred.shape == (feat.shape[0], 2))
        assert (quat_pred.shape == (feat.shape[0], 4))

        return torch.cat((scale_pred, trans_pred, quat_pred), dim=1)


class ScaleNet(nn.Module):
    def __init__(self, nz, bias=0.75):
        super(ScaleNet, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)
        self.bias = bias

    def forward(self, feat):
        scale = self.pred_layer(feat) + self.bias  # biasing the scale
        scale = torch.nn.functional.relu(scale) + 1e-12
        # print(self.bias, scale.squeeze(), '\n')
        # print('scale: ( Mean = {}, Var = {} )'.format(scale.mean().item(), scale.var().item()))
        return scale


class TransNet(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """
    def __init__(self, nz, orth=True):
        super(TransNet, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class QuatNet(nn.Module):
    def __init__(self, nz_feat, nz_rot=4):
        super(QuatNet, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)

    def forward(self, feat):
        quat = self.pred_layer(feat)
        quat = torch.nn.functional.normalize(quat)
        return quat


class RotationNet(nn.Module):
    """
    This class linearly maps deep features into a rotation matrix with dimension set by nz_rot.
    """
    def __init__(self, nz_feat, nz_rot=9):
        super(RotationNet, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)

    def forward(self, feat):
        rot = self.pred_layer(feat)
        # rot = torch.nn.functional.normalize(rot)
        return rot

